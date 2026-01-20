# 일단 1층만 만들기
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self,x_dim,c_dim,z_dim=16,h1 = 32,h2 =64):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        # encoder
        self.enc1 = nn.Linear(x_dim+c_dim,h1)
        self.enc2 = nn.Linear(h1,h2)
        self.mu = nn.Linear(h2,z_dim)
        self.logvar = nn.Linear(h2,z_dim)
        # decoder
        self.dec1 = nn.Linear(z_dim+c_dim,h2) # 16+9 ,64
        self.dec2 = nn.Linear(h2,h1) # 64,32
        self.out = nn.Linear(h1,x_dim) # 32 23
    def encoder(self,x,c):
        h = torch.cat([x,c],dim = 1)
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
        # reparameterize
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        # decoder
    def decoder(self,z,c):
        h = torch.cat([z,c],dim = 1)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        x_hat = self.out(h)
        return x_hat
    def forward(self,x,c):
        mu,logvar = self.encoder(x,c)
        z = self.reparameterize(mu,logvar)
        x_hat = self.decoder(z,c)
        return x_hat,mu,logvar

def cvae_loss_by_nickel_presence(
    x_hat, x, mu, logvar,
    beta=1.0,
    nickel_weight=0.8,   # 니켈 항목에 주는 비중(0~1)
    ni_idx=7,
    zero_thr=1e-6
):
    """
    반환:
      - loss_no_ni: 니켈 없는 샘플(ni==0)에 대한 loss
      - loss_has_ni: 니켈 있는 샘플(ni>0)에 대한 loss (니켈 가중치 포함)
      - loss_total: 전체 loss (둘을 샘플수로 가중 평균)
      - 로그용 recon/kl 등
    """

    # --- 마스크: 니켈 유무 ---
    ni_true = x[:, ni_idx]                         # [B]
    mask_no_ni  = (ni_true <= zero_thr)            # [B]
    mask_has_ni = ~mask_no_ni                      # [B]

    # --- 공통: KL(배치 평균) ---
    kl_vec = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]

    # --- feature 인덱스: 니켈 제외 ---
    idx_no_ni_feat = list(range(0, ni_idx)) + list(range(ni_idx + 1, x.size(1)))

    if mask_no_ni.any():
        x0_hat = x_hat[mask_no_ni]
        x0     = x[mask_no_ni]
        kl0    = kl_vec[mask_no_ni].mean()

        recon0_all = F.mse_loss(x0_hat, x0, reduction='mean')
        recon0_no_ni = F.mse_loss(x0_hat[:, idx_no_ni_feat], x0[:, idx_no_ni_feat], reduction='mean')
        recon0_ni = F.mse_loss(x0_hat[:, ni_idx:ni_idx+1], x0[:, ni_idx:ni_idx+1], reduction='mean')

        # 니켈 없는 경우는 "니켈을 0으로 붙이기"가 목적이므로
        # 니켈 항목 recon을 더 주고 싶으면 가중치를 주면 됨(예: nickel_weight를 여기에도 활용 가능)
        recon0 = (1 - nickel_weight) * recon0_no_ni + nickel_weight * recon0_ni

        loss_no_ni = recon0 + beta * kl0
    else:
        loss_no_ni = torch.tensor(0.0, device=x.device)
        recon0_all = torch.tensor(0.0, device=x.device)
        recon0_no_ni = torch.tensor(0.0, device=x.device)
        recon0_ni = torch.tensor(0.0, device=x.device)
        kl0 = torch.tensor(0.0, device=x.device)

    if mask_has_ni.any():
        x1_hat = x_hat[mask_has_ni]
        x1     = x[mask_has_ni]
        kl1    = kl_vec[mask_has_ni].mean()

        recon1_all = F.mse_loss(x1_hat, x1, reduction='mean')
        recon1_no_ni = F.mse_loss(x1_hat[:, idx_no_ni_feat], x1[:, idx_no_ni_feat], reduction='mean')
        recon1_ni = F.mse_loss(x1_hat[:, ni_idx:ni_idx+1], x1[:, ni_idx:ni_idx+1], reduction='mean')

        # 니켈 있는 경우: 니켈 항목을 더 크게 반영
        recon_has_ni = (1 - nickel_weight) * recon1_no_ni + nickel_weight * recon1_ni

        loss_has_ni = recon_has_ni + beta * kl1
    else:
        loss_has_ni = torch.tensor(0.0, device=x.device)
        recon1_all = torch.tensor(0.0, device=x.device)
        recon1_no_ni = torch.tensor(0.0, device=x.device)
        recon1_ni = torch.tensor(0.0, device=x.device)
        kl1 = torch.tensor(0.0, device=x.device)

    # ========== (C) 전체 loss: 샘플 수로 가중 평균 ==========
    n0 = mask_no_ni.sum().item()
    n1 = mask_has_ni.sum().item()
    if n0 + n1 > 0:
        loss_total = (loss_no_ni * n0 + loss_has_ni * n1) / (n0 + n1)
    else:
        loss_total = torch.tensor(0.0, device=x.device)

    # 로그 반환
    logs = {
        "n_no_ni": n0,
        "n_has_ni": n1,
        "loss_no_ni": loss_no_ni.detach(),
        "loss_has_ni": loss_has_ni.detach(),
        "loss_total": loss_total.detach(),

        "recon_no_ni_all": recon0_all.detach(),
        "recon_no_ni_except": recon0_no_ni.detach(),
        "recon_no_ni_ni": recon0_ni.detach(),
        "kl_no_ni": kl0.detach(),

        "recon_has_ni_all": recon1_all.detach(),
        "recon_has_ni_except": recon1_no_ni.detach(),
        "recon_has_ni_ni": recon1_ni.detach(),
        "kl_has_ni": kl1.detach(),
    }

    return loss_total, loss_no_ni, loss_has_ni, logs
