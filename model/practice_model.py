import torch
import torch.nn
import torch.nn.functional as F
class MultiDecoderCondVAE(nn.Module):
    def __init__(self,x_dim,c_dim,h1=32,h2=64,z_dim=16):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h2,z_dim)
        self.logvar_head = nn.Linear(h2,z_dim)
        
        ## bce_encoder
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim)
        )
        ## mse_decoder
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim+c_dim+x_dim,128),
            nn.ReLU(),
            nn.Linear(128,x_dim)
        )

    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu+std*eps
    
    def forward(self,x,c):
        z_mu = self.mu_head(self.encoder(torch.cat([x,c],dim = 1)))
        z_logvar =self.logvar_head(self.encoder(torch.cat([x,c],dim = 1)))
        z = self.reparameterize(z_mu,z_logvar)

        mask_logits= self.decoder_bce(torch.cat([z,c],dim = 1))
        prob_mask = torch.sigmoid(mask_logits)
        recon_numeric = self.decoder_mse(torch.cat([z,c,prob_mask],dim =1 ))
        final_recon = recon_numeric*prob_mask
        return mask_logits, recon_numeric, z_mu,z_logvar
    def intergrated_loss_fn(mask_logits,recon_numeric,target_x,mu,logvar,beta = 0.1, lam=1.0):