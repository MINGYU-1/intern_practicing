import torch
import torch.nn as nn
import torch.nn.functional as F

def cvae_loss3_general(x_hat, x, mu, logvar, beta=1.0, focus_idx=None, focus_weight=1.0):
    """
    loss3 = recon_all (전체 MSE) + focus_weight * recon_focus (특정 인덱스 MSE) + beta * KL
    """
    # 1. 전체 Reconstruction Loss
    recon_all = F.mse_loss(x_hat, x, reduction="mean")

    # 2. 특정 인덱스(Focus) Reconstruction Loss
    recon_focus = torch.tensor(0.0, device=x.device)
    if focus_idx is not None:
        if isinstance(focus_idx, int):
            focus_idx = [focus_idx]
        
        idx_tensor = torch.tensor(focus_idx, device=x.device)
        # index_select를 사용하여 특정 feature(열)만 추출
        x_hat_focus = x_hat.index_select(1, idx_tensor)
        x_focus = x.index_select(1, idx_tensor)
        recon_focus = F.mse_loss(x_hat_focus, x_focus, reduction="mean")

    # 3. KL Divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # 최종 조합
    recon3 = recon_all + (focus_weight * recon_focus)
    loss3 = recon3 + (beta * kl)
    
    return loss3, recon3, recon_all, recon_focus, kl

class CVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=16, h1=32, h2=64):
        super().__init__()
        self.z_dim = z_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + c_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU()
        )
        self.mu = nn.Linear(h2, z_dim)
        self.logvar = nn.Linear(h2, z_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + c_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, x_dim)
        )

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar

class CascadeCVAE(nn.Module):
    def __init__(self, metal_dim, supporting_dim, pretreat_dim, operating_dim, z_dim=16, h1=32, h2=64):
        super().__init__()
        self.stage1 = CVAE(metal_dim, operating_dim, z_dim, h1, h2)
        self.stage2 = CVAE(supporting_dim, operating_dim + metal_dim, z_dim, h1, h2)
        self.stage3 = CVAE(pretreat_dim, operating_dim + metal_dim + supporting_dim, z_dim, h1, h2)

    def forward(self, metal, supporting, pretreat, operating):
        m_hat, m_mu, m_logvar = self.stage1(metal, operating)
        s_hat, s_mu, s_logvar = self.stage2(supporting, torch.cat([operating, metal], dim=1))
        p_hat, p_mu, p_logvar = self.stage3(pretreat, torch.cat([operating, metal, supporting], dim=1))

        return {
            "stage1": {"x_hat": m_hat, "mu": m_mu, "logvar": m_logvar},
            "stage2": {"x_hat": s_hat, "mu": s_mu, "logvar": s_logvar},
            "stage3": {"x_hat": p_hat, "mu": p_mu, "logvar": p_logvar},
        }

    def compute_loss_all_loss3(
        self, out_dict, metal, supporting, pretreat,
        beta=1.0, 
        metal_focus_idx=7, metal_focus_weight=1.0  # metal의 7번 인덱스 강조
    ):
        # Stage 1 (Metal) - focus_idx=7 적용
        loss1, r3_1, ra_1, rf_1, kl1 = cvae_loss3_general(
            out_dict["stage1"]["x_hat"], metal, 
            out_dict["stage1"]["mu"], out_dict["stage1"]["logvar"],
            beta=beta, focus_idx=metal_focus_idx, focus_weight=metal_focus_weight
        )

        # Stage 2 (Supporting)
        loss2, r3_2, ra_2, rf_2, kl2 = cvae_loss3_general(
            out_dict["stage2"]["x_hat"], supporting, 
            out_dict["stage2"]["mu"], out_dict["stage2"]["logvar"],
            beta=beta, focus_idx=None
        )

        # Stage 3 (Pretreat)
        loss3, r3_3, ra_3, rf_3, kl3 = cvae_loss3_general(
            out_dict["stage3"]["x_hat"], pretreat, 
            out_dict["stage3"]["mu"], out_dict["stage3"]["logvar"],
            beta=beta, focus_idx=None
        )

        total_loss = loss1 + loss2 + loss3
        return total_loss, {"total": total_loss, "s1_recon_focus": rf_1, "s1_kl": kl1}