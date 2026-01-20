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

def cvae_loss_masked(x_hat, x, mu, logvar, beta=1.0, nonzero_weight=0.8):
    """
    금속 성분 유무(0 vs non-zero)를 구분하여 Loss를 계산하는 함수
    
    Args:
        x_hat: 모델의 복원 결과 [batch_size, 23]
        x: 실제 입력값 [batch_size, 23]
        nonzero_weight: 0이 아닌 실제 성분에 부여할 가중치 (나머지는 1-nonzero_weight)
    """
    # 1. 마스크 생성 (0보다 큰 위치 찾기)
    mask_nonzero = (x > 0).float()
    mask_zero = (x == 0).float()
    
    # 2. 개별 요소별 제곱 오차 계산
    diff_sq = (x_hat - x) ** 2
    
    # 3. 0이 아닌 부분의 MSE (Non-zero 항목 평균)
    # 0으로 나누는 것을 방지하기 위해 epsilon(1e-8) 추가
    recon_nonzero = (diff_sq * mask_nonzero).sum() / (mask_nonzero.sum() + 1e-8)
    
    # 4. 0인 부분의 MSE (Zero 항목 평균)
    recon_zero = (diff_sq * mask_zero).sum() / (mask_zero.sum() + 1e-8)
    
    # 5. 가중치가 적용된 Reconstruction Loss
    # nonzero_weight가 0.8이면 실제 성분 오차에 80%, 0인 부분 오차에 20% 비중 부여
    recon_weighted = (nonzero_weight * recon_nonzero) + ((1 - nonzero_weight) * recon_zero)
    
    # 6. KL Divergence (배치 평균)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # 최종 손실 함수
    loss_total = recon_weighted + beta * kl
    
    # 전체 MSE (모니터링용)
    recon_all = diff_sq.mean()
    
    return loss_total, recon_all, recon_nonzero, recon_zero, kl