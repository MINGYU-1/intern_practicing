import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedHurdleCVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=16, h1_dim=64, h2_dim=32):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        # --- Encoder: (x+c) -> h1 -> h2 ---
        self.enc_fc1 = nn.Linear(x_dim + c_dim, h1_dim)
        self.enc_bn1 = nn.BatchNorm1d(h1_dim)
        self.enc_fc2 = nn.Linear(h1_dim, h2_dim)

        self.mu = nn.Linear(h2_dim, z_dim)
        self.logvar = nn.Linear(h2_dim, z_dim)

        # --- Decoder: (z+c) -> h2 -> h1 ---
        self.dec_fc1 = nn.Linear(z_dim + c_dim, h2_dim)
        self.dec_fc2 = nn.Linear(h2_dim, h1_dim)

        # residual block MUST keep same dim (h1_dim)
        self.res_block = nn.Sequential(
            nn.Linear(h1_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h1_dim)
        )

        # heads are produced from h1_dim
        self.pres_logits = nn.Linear(h1_dim, x_dim)
        self.amt_mu = nn.Linear(h1_dim, x_dim)
        self.amt_logvar = nn.Linear(h1_dim, x_dim)

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.enc_fc1(h)
        h = self.enc_bn1(h)
        h = F.relu(h)
        h = F.relu(self.enc_fc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = F.relu(self.dec_fc1(h))      # (B, h2_dim)
        h = F.relu(self.dec_fc2(h))      # (B, h1_dim)

        # residual at h1_dim
        h = F.relu(h + self.res_block(h))

        p_logits = self.pres_logits(h)
        mu_log = torch.clamp(self.amt_mu(h), min=-10, max=10)
        logvar_log = torch.clamp(self.amt_logvar(h), min=-7, max=2)

        return p_logits, mu_log, logvar_log

    def forward(self, x, c):
        mu_z, logvar_z = self.encode(x, c)
        z = self.reparameterize(mu_z, logvar_z)
        p_logits, mu_log, logvar_log = self.decode(z, c)
        return p_logits, mu_log, logvar_log, mu_z, logvar_z



def advanced_hurdle_loss(p_logits, mu_log, logvar_log, x, mu_z, logvar_z,
                         beta=1.0, alpha=1.0, lambda_reg=100.0, eps=1e-7):
    y = (x > 0).float()
    mask_pos = y

    # 1) Presence BCE
    bce = F.binary_cross_entropy_with_logits(p_logits, y, reduction='mean')

    # 2) Amount LogNormal NLL (only for positive)
    clamped_logvar = torch.clamp(logvar_log, min=-7, max=2)
    var_log = torch.exp(clamped_logvar)

    logx = torch.log(torch.clamp(x, min=eps))

    nll_elem = 0.5 * (((logx - mu_log) ** 2) / (var_log + eps) + clamped_logvar + 1.837)
    pos_loss = (nll_elem * mask_pos).sum() / (mask_pos.sum() + eps)

    # 3) Huber in log-domain (only for positive)
    huber_loss = F.smooth_l1_loss(mu_log * mask_pos, logx * mask_pos, reduction='sum') / (mask_pos.sum() + eps)

    # 4) KL
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - torch.exp(logvar_z), dim=1).mean()

    total_loss = (alpha * bce) + pos_loss + (lambda_reg * huber_loss) + (beta * kl_loss)

    return {"loss": total_loss, "bce": bce, "pos": pos_loss, "mse": huber_loss, "kl": kl_loss}
