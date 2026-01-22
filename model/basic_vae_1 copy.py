import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,x_dim,c_dim,h1_dim,h2_dim,z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim

    def forward(self,x,c):
        menas = self.linear_means(x)
        return means, log_vars

def masked_mse(recon, target, mask, eps=1e-8):
    num = ((recon - target) ** 2 * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def multi_loss_fn(prob_mask, recon_numeric, final_recon,
                  target_bce, target_mse,
                  lam_bce=1.0, lam_num=1.0, lam_final=0.0):
    bce = F.binary_cross_entropy(prob_mask, target_bce)

    num_mse = masked_mse(recon_numeric, target_mse, target_bce)

    final_mse = masked_mse(final_recon, target_mse, target_bce)

    return lam_bce*bce + lam_num*num_mse + lam_final*final_mse
