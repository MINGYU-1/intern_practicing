import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE2(nn.Module):
   
    def __init__(self, s_dim, z1_dim=16, z2_dim=16, h1=32, h2=64):
        super().__init__()
        self.s_dim = s_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim

        self.enc1 = nn.Linear(z1_dim + s_dim, h1) ##z1에 구현
        self.enc2 = nn.Linear(h1, h2)
        self.mu = nn.Linear(h2, z_dim)
        self.logvar = nn.Linear(h2, z_dim)

        # Decoder: [z, c, s] -> h2 -> [h2, s] -> h1 -> x_hat
        self.dec1 = nn.Linear(z_dim + c_dim + s_dim, h2)
        self.dec2 = nn.Linear(h2 + s_dim, h1)
        self.out = nn.Linear(h1, x_dim)

    def encoder(self, x, c, s):
        h = torch.cat([x, c, s], dim=1)
        h = F.relu(self.enc1(h))
        h = torch.cat([h, s], dim=1)   # support 재주입
        h = F.relu(self.enc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z, c, s):
        h = torch.cat([z, c, s], dim=1)
        h = F.relu(self.dec1(h))
        h = torch.cat([h, s], dim=1)   # support 재주입
        h = F.relu(self.dec2(h))
        x_hat = self.out(h)
        return x_hat

    def forward(self, x, c, s):
        mu, logvar = self.encoder(x, c, s)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c, s)
        return x_hat, mu, logvar
