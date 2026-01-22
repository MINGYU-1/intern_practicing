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

        # Decoder 1 (mask): [z, c] -> prob_mask (x_dim)
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim + c_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x_dim),
        )

        # Decoder 2 (numeric): [z, c, prob_mask] -> recon_numeric (x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim + c_dim + x_dim, 128),
            nn.ReLU(),
            nn.Linear(128,x_dim)
        )

    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu+eps*std

    def forward(self, x, c):
        z_mu = self.mu_head(self.encoder(torch.cat([x, c], dim=1)))
        z_logvar = self.logvar_head(self.encoder(torch.cat([x, c], dim=1)))
        z = self.reparameterize(z_mu, z_logvar)
        
        mask_logits = self.decoder_bce(torch.cat([z, c], dim=1))
        prob_mask = torch.sigmoid(mask_logits)
        recon_numeric = self.decoder_mse(torch.cat([z, c, prob_mask], dim=1))
        final_recon = recon_numeric * prob_mask
        return mask_logits, recon_numeric, z_mu, z_logvar

def integrated_loss_fn(mask_logits, recon_numeric, target_x, mu, logvar, beta=1.0,lam = 1.0):

    target_mask = (target_x > 1e-8).float() # target_x의 값이 미소한 1e-6보다 크게 채택하여 
    ##**with_logits_사용이유**: target_mask로 해서 수행하기 위해서 
    bce_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask, reduction='sum') #확실하게 0,1로 구분
 

    # 2. Gaussian Value Loss (Masked MSE)
    # 실제 값이 존재하는 영역만 MSE 계산
    mse_elements = (recon_numeric - target_x) ** 2
    masked_mse_loss = torch.sum(mse_elements * target_mask) 

    # 3. KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 스케일 조정을 위해 배치 사이즈로 나누는 것이 안정적입니다.
    batch_size = target_x.size(0)

    total_loss = (bce_loss + masked_mse_loss  + beta * kl_loss) / batch_size

    
    return {
        'loss': total_loss,
        'bce': bce_loss / batch_size,
        'mse': masked_mse_loss / batch_size,
        'kl': kl_loss / batch_size
    }