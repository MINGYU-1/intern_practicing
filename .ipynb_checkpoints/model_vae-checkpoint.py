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
        
def cvae_loss(x_hat, x, mu, logvar, beta=1.0):
    # recon: 배치 평균 MSE
    recon = F.mse_loss(x_hat, x)

    # KL: 배치 평균
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl.mean()

    loss = recon + beta * kl
    return loss, recon, kl
def cvae_loss_each(x_hat,x,mu,logvar,beta = 0.01):
    losses=[]
    recons=[]
    kls = []
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl.mean()
    for i in range(x_hat.shape[1]):
        recon = F.mse_loss(x_hat[:,i],x[:,i])
        loss = recon+beta*kl
        losses.append(loss)
        recons.append(recon)
        kls.append(kl)
    return losses,recons,kls
def cvae_loss_optimized(x_hat, x, mu, logvar, beta=0.01):
    recon_each_feature = F.mse_loss(x_hat, x, reduction='none').mean(dim=0)
    
    # 2. KL Divergence 계산
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    # 3. 각 피처별 최종 Loss 리스트 생성
    losses = [r + beta * kl for r in recon_each_feature]
    
    return losses, recon_each_feature.tolist(), [kl.item()] * len(losses)