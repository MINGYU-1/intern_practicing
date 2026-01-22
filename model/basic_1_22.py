import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDecoderCondVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=16, h1=32, h2=64):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        # Encoder: [x, c] -> z
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + c_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(h2,z_dim) ## 평균
        self.logvar_head = nn.Linear(h2,z_dim) ## 분산

        # Decoder 1 (mask): [z, x] -> prob_mask (x_dim)
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim + x_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x_dim),
        )

        # Decoder 2 (numeric): [z, c] -> recon_numeric (x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim + c_dim, 128),
            nn.ReLU(),
            nn.Linear(128,x_dim)
        )

    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu+eps*std

    def forward(self, x, c, hard_mask=False):
        h = self.encoder(torch.cat([x, c], dim=1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu, z_logvar)

        mask_logits = self.decoder_bce(torch.cat([z, x], dim=1))
        prob_mask = torch.sigmoid(mask_logits)
        mask_out = (mask_logits > 0).float()  # 0/1 출력

        recon_numeric = self.decoder_mse(torch.cat([z, c], dim=1))

        return mask_logits, prob_mask, mask_out, recon_numeric, z_mu, z_logvar

def integrated_loss_fn(mask_logits, recon_numeric, target_x, mu, logvar, beta=1.0):
    # 1. Target Mask 생성 (0보다 크면 존재함=1, 아니면 0)
    target_mask = (target_x > 0).float()
    
    # 2. BCE Loss: 존재 유무 학습 (가장 중요!)
    bce_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask, reduction='sum')
    
    # 3. Masked MSE: 실제 값이 있는 지점만 수치 학습
    mse_elements = (recon_numeric - target_x) ** 2
    masked_mse_loss = torch.sum(mse_elements * target_mask) 
    
    # 4. KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    batch_size = target_x.size(0)
    total_loss = (bce_loss + beta * kl_loss) / batch_size

    return {
        'loss': total_loss,
        'bce': bce_loss / batch_size,
        'mse': masked_mse_loss / batch_size,
        'kl': kl_loss / batch_size
    }