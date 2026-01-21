import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedHurdleCVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=16, h1_dim=128, h2_dim=64): # 차원 상향
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        # Encoder: BN 대신 LayerNorm이 안정적일 때가 많습니다.
        self.enc_fc = nn.Sequential(
            nn.Linear(x_dim + c_dim, h1_dim),
            nn.LayerNorm(h1_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h1_dim, h2_dim),
            nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(h2_dim, z_dim)
        self.logvar = nn.Linear(h2_dim, z_dim)

        # Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(z_dim + c_dim, h2_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h2_dim, h1_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Residual Block 강화
        self.res_block = nn.Sequential(
            nn.Linear(h1_dim, h1_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h1_dim, h1_dim)
        )

        self.pres_logits = nn.Linear(h1_dim, x_dim)
        self.amt_mu = nn.Linear(h1_dim, x_dim)
        self.amt_logvar = nn.Linear(h1_dim, x_dim)

    def encode(self, x, c):
        h = self.enc_fc(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        h = self.dec_fc(torch.cat([z, c], dim=1))
        h = h + self.res_block(h)

        p_logits = self.pres_logits(h)
        # 중요: mu_log를 바로 쓰지 않고, 어느 정도 범위를 강제
        mu_log = self.amt_mu(h) 
        logvar_log = torch.clamp(self.amt_logvar(h), -5, 2)

        return p_logits, mu_log, logvar_log

    def forward(self, x, c):
        mu_z, logvar_z = self.encode(x, c)
        z = self.reparameterize(mu_z, logvar_z)
        return (*self.decode(z, c), mu_z, logvar_z)

    @torch.no_grad()
    def predict_mean(self, z, c, threshold=0.5):
        p_logits, mu_log, logvar_log = self.decode(z, c)
        p = torch.sigmoid(p_logits)
        
        # Hard Gating: 확률이 낮으면 가차없이 0으로 밀어버림 (그래프의 노이즈 제거 핵심)
        mask = (p > threshold).float()
        
        # Log-Normal Mean
        amt_mean = torch.exp(mu_log + 0.5 * torch.exp(logvar_log))
        return mask * amt_mean

def advanced_hurdle_loss(p_logits, mu_log, logvar_log, x, mu_z, logvar_z, 
                         alpha=5.0, beta=1.0, lambda_reg=100.0, eps=1e-6):
    y = (x > 0).float()
    mask_pos = y
    mask_neg = 1.0 - y

    # 1) Presence (BCE)
    # pos_weight를 tensor로 직접 선언할 때 device를 확실히 맞춤
    pos_weight = torch.ones([p_logits.shape[1]], device=p_logits.device) * 10.0
    bce = F.binary_cross_entropy_with_logits(p_logits, y, pos_weight=pos_weight)

    # 2) Amount (Huber)
    # x가 0인 곳에서 log가 계산되지 않도록 아주 작은 값을 더함
    # clamp를 사용해 log(0)을 원천 차단
    logx = torch.log(torch.clamp(x, min=eps))
    
    # Positive 샘플이 하나도 없는 배치일 경우를 대비
    pos_count = mask_pos.sum()
    if pos_count > 0:
        huber_loss = F.smooth_l1_loss(mu_log * mask_pos, logx * mask_pos, reduction='sum') / (pos_count + eps)
    else:
        huber_loss = torch.tensor(0.0, device=x.device)

    # 3) Negative Constraint (노이즈 제거)
    # mu_log가 폭주하지 않도록 clamp 후 계산
    mu_log_clamped = torch.clamp(mu_log, max=20) 
    neg_penalty = torch.relu(mu_log_clamped + 10).pow(2) * mask_neg
    neg_loss = neg_penalty.sum() / (mask_neg.sum() + eps)

    # 4) Latent KL
    # KL이 폭발하지 않도록 logvar_z를 clamp
    logvar_z = torch.clamp(logvar_z, -10, 10)
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - torch.exp(logvar_z), dim=1).mean()

    # 각 항목이 nan인지 체크 (디버깅용)
    total_loss = (alpha * bce) + (lambda_reg * huber_loss) + (beta * kl_loss) + (alpha * neg_loss)

    return {"loss": total_loss, "bce": bce, "huber": huber_loss, "kl": kl_loss, "neg": neg_loss}