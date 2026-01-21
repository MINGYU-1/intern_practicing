import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdvancedHurdleCVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=16, h_dim=64):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        # --- Encoder ---
        self.enc_input = nn.Sequential(
            nn.Linear(x_dim + c_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

        # --- Decoder ---
        self.dec_input = nn.Sequential(
            nn.Linear(z_dim + c_dim, h_dim),
            nn.ReLU()
        )
        
        self.res_block = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )

        self.pres_logits = nn.Linear(h_dim, x_dim)
        self.amt_mu = nn.Linear(h_dim, x_dim)
        self.amt_logvar = nn.Linear(h_dim, x_dim)

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.enc_input(h)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # new_hurdle_model1.py 내 decode 함수 수정
    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = self.dec_input(h)
        h = F.relu(h + self.res_block(h))
        
        p_logits = self.pres_logits(h)
        
        mu_log = torch.clamp(self.amt_mu(h), min=-10, max=10) 
        logvar_log = torch.clamp(self.amt_logvar(h), min=-7, max=2)
        
        return p_logits, mu_log, logvar_log

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        p_logits, mu_log, logvar_log = self.decode(z, c)
        return p_logits, mu_log, logvar_log, mu, logvar

def advanced_hurdle_loss(p_logits, mu_log, logvar_log, x, mu_z, logvar_z, 
                         beta=1.0, alpha=1.0, lambda_reg=100.0, eps=1e-7):
    y = (x > 0).float()
    mask_pos = y
    
    # 1. Presence (BCE)
    bce = F.binary_cross_entropy_with_logits(p_logits, y, reduction='mean')
    
    # 2. Amount (LogNormal NLL) - x가 이미 로그 스케일인지 확인 필요!
    # 만약 x가 원본이라면 log(x)를 해야하지만, x가 이미 전처리된(StandardScaled) 상태라면 
    # 아래 logx = x로 처리해야 합니다. (여기서는 원본이라 가정 시)
    clamped_logvar = torch.clamp(logvar_log, min=-7, max=2)
    var_log = torch.exp(clamped_logvar)
    
    # 안전한 로그 계산 (x가 원본 스케일일 때만 log를 취함)
    # x가 StandardScaler를 거친 데이터라면 아래 logx = x 로 수정하세요.
    logx = torch.log(torch.clamp(x, min=eps)) 
    
    nll_elem = 0.5 * (((logx - mu_log) ** 2) / (var_log + eps) + clamped_logvar + 1.837)
    pos_loss = (nll_elem * mask_pos).sum() / (mask_pos.sum() + eps)
    
    # 3. Huber Loss (폭발 방지를 위해 mu_log에 직접 걸거나 exp를 제한)
    # 방법 A: 로그 도메인에서 직접 MSE/Huber 적용 (권장)
    huber_loss = F.smooth_l1_loss(mu_log * mask_pos, logx * mask_pos, reduction='sum') / (mask_pos.sum() + eps)
    
    # 만약 꼭 원본 스케일 기댓값으로 계산해야 한다면 mu_log를 강하게 제한:
    # pred_mean = torch.exp(torch.clamp(mu_log + 0.5 * var_log, max=10)) 
    
    # 4. KL
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - torch.exp(logvar_z), dim=1).mean()
    
    total_loss = (alpha * bce) + pos_loss + (lambda_reg * huber_loss) + (beta * kl_loss)
    
    # MSE 기록용 (history dict에 'mse'가 있으므로 'huber' 대신 'mse'로 반환)
    return {"loss": total_loss, "bce": bce, "pos": pos_loss, "mse": huber_loss, "kl": kl_loss}