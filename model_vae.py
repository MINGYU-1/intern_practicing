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
        x_hat = torch.sigmoid(self.out(h))
        return x_hat
    def forward(self,x,c):
        mu,logvar = self.encoder(x,c)
        z = self.reparameterize(mu,logvar)
        x_hat = self.decoder(z,c)
        return x_hat,mu,logvar
        
def cvae_loss(x_hat, x, mu, logvar, beta=1.0, nickel_idx=7, nickel_weight=1.0,
              logvar_clip=(-10.0, 10.0)):
    """
    recon1: 전체 MSE (다 포함)
    recon2: 니켈 제외 MSE
    recon3: 니켈함량(수정방식) = recon2 + nickel_weight * recon4
            (즉, 니켈 제외 + 니켈을 명시적으로 가중합)
    recon4: 니켈만 MSE
    """

    # ---- 안전성 체크 ----
    if x_hat.shape != x.shape:
        raise ValueError(f"x_hat shape {x_hat.shape} != x shape {x.shape}")
    if mu.shape != logvar.shape:
        raise ValueError(f"mu shape {mu.shape} != logvar shape {logvar.shape}")

    B, D = x.shape
    device = x.device

    # 니켈 제외 인덱스
    other_idx = torch.tensor([i for i in range(D) if i != nickel_idx],
                             device=device, dtype=torch.long)

    # ---- recon ----
    recon1 = F.mse_loss(x_hat, x, reduction="mean")  # 다 포함

    x_hat_others = x_hat.index_select(1, other_idx)
    x_others     = x.index_select(1, other_idx)
    recon2 = F.mse_loss(x_hat_others, x_others, reduction="mean")  # 니켈 제외

    recon4 = F.mse_loss(x_hat[:, nickel_idx], x[:, nickel_idx], reduction="mean")  # 니켈만

    # 니켈함량(수정방식): 교차항 없이 명시적 가중합
    recon3 = recon2 + nickel_weight * recon4

    # ---- KL ----
    if logvar_clip is not None:
        logvar = logvar.clamp(*logvar_clip)

    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl.mean()

    # ---- total loss ----
    loss1 = recon1 + beta * kl
    loss2 = recon2 + beta * kl
    loss3 = recon3 + beta * kl
    loss4 = recon4 + beta * kl

    return loss1, loss2, loss3, loss4, recon1, recon2, recon3, recon4, kl