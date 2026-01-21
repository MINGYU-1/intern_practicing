import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedMultiHurdleCVAE(nn.Module):
    def __init__(self, x_dim, c_dim, num_classes, z_dim=16, h1_dim=64, h2_dim=32):
        super().__init__()
        self.x_dim = x_dim
        self.num_classes = num_classes # 예: 0(없음), 1(소량), 2(대량) 등

        # --- Encoder ---
        self.enc_fc1 = nn.Linear(x_dim + c_dim, h1_dim)
        self.enc_bn1 = nn.BatchNorm1d(h1_dim)
        self.enc_fc2 = nn.Linear(h1_dim, h2_dim)
        self.mu = nn.Linear(h2_dim, z_dim)
        self.logvar = nn.Linear(h2_dim, z_dim)

        # --- Decoder ---
        self.dec_fc1 = nn.Linear(z_dim + c_dim, h2_dim)
        self.dec_fc2 = nn.Linear(h2_dim, h1_dim)

        self.res_block = nn.Sequential(
            nn.Linear(h1_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h1_dim)
        )

        # Multi-class Classification Head: (Batch, x_dim * num_classes)
        self.class_logits = nn.Linear(h1_dim, x_dim * num_classes)
        
        # Continuous Amount Head (Gaussian/MSE): (Batch, x_dim)
        self.amt_mu = nn.Linear(h1_dim, x_dim)

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = F.relu(self.enc_bn1(self.enc_fc1(h)))
        h = F.relu(self.enc_fc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = F.relu(self.dec_fc1(h))
        h = F.relu(self.dec_fc2(h))
        h = F.relu(h + self.res_block(h))

        # Class logits reshape: (B, x_dim, num_classes)
        logits = self.class_logits(h).view(-1, self.x_dim, self.num_classes)
        mu_val = self.amt_mu(h)
        
        return logits, mu_val

    def forward(self, x, c):
        mu_z, logvar_z = self.encode(x, c)
        z = self.reparameterize(mu_z, logvar_z)
        logits, mu_val = self.decode(z, c)
        return logits, mu_val, mu_z, logvar_z
def multi_hurdle_loss(logits, mu_val, x, target_classes, mu_z, logvar_z, beta=1.0):
    """
    Args:
        logits: (B, x_dim, num_classes) - 분류 예측
        mu_val: (B, x_dim) - 수치 예측
        x: (B, x_dim) - 실제 수치 target
        target_classes: (B, x_dim) - 실제 클래스 target (0, 1, 2...)
    """
    B, x_dim, num_classes = logits.shape

    # 1) Classification Loss (CCE)
    # CrossEntropyLoss는 (N, C, d1, d2...) 형태를 받으므로 차원 재배치
    logits_flat = logits.permute(0, 2, 1) # (B, num_classes, x_dim)
    cce_loss = F.cross_entropy(logits_flat, target_classes, reduction='mean')

    # 2) Regression Loss (MSE) - 오직 양수(클래스 > 0)인 샘플에 대해서만 계산
    mask = (target_classes > 0).float()
    mse_elem = F.mse_loss(mu_val, x, reduction='none')
    mse_loss = (mse_elem * mask).sum() / (mask.sum() + 1e-7)

    # 3) KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - torch.exp(logvar_z), dim=1).mean()

    total_loss = cce_loss + mse_loss + (beta * kl_loss)

    return {
        "loss": total_loss,
        "cce": cce_loss,
        "mse": mse_loss,
        "kl": kl_loss
    }