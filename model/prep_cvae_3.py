import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE_Prep(nn.Module):
    """
    CVAE with:
      - support(s) + preparation(p) conditioning
      - dual injection (re-inject s and p inside hidden layers)
      - auxiliary head to predict x_preparation (p_hat)

    Inputs:
      x: (batch, x_dim)
      c: (batch, c_dim)
      s: (batch, s_dim)
      p: (batch, p_dim)  # preparation vector

    Outputs:
      x_hat: (batch, x_dim)
      p_hat: (batch, p_dim)
      mu, logvar: (batch, z_dim)
    """
    def __init__(self, x_dim, c_dim, s_dim, p_dim, z_dim=16, h1=32, h2=64):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.s_dim = s_dim
        self.p_dim = p_dim
        self.z_dim = z_dim

        # ---------- Encoder ----------
        # preparation(p)을 "마지막 조건"으로 넣되, 실제로는 concat 순서만 마지막으로 두면 된다.
        self.enc1 = nn.Linear(x_dim + c_dim + s_dim + p_dim, h1)
        self.enc2 = nn.Linear(h1 + s_dim + p_dim, h2)
        self.mu = nn.Linear(h2, z_dim)
        self.logvar = nn.Linear(h2, z_dim)

        # ---------- Decoder ----------
        self.dec1 = nn.Linear(z_dim + c_dim + s_dim + p_dim, h2)
        self.dec2 = nn.Linear(h2 + s_dim + p_dim, h1)
        self.out_x = nn.Linear(h1, x_dim)

        # ---------- Preparation prediction head ----------
        # "x_preparation을 구한다" = p를 모델이 예측하도록 보조헤드를 둠.
        # mu 기반으로 예측하는 게 안정적이다.
        self.out_p = nn.Sequential(
            nn.Linear(z_dim + c_dim + s_dim, 64),
            nn.ReLU(),
            nn.Linear(64, p_dim)
        )

    def encoder(self, x, c, s, p):
        # preparation(p)을 마지막에 concat
        h = torch.cat([x, c, s, p], dim=1)
        h = F.relu(self.enc1(h))
        # 은닉층에서도 s, p 재주입
        h = torch.cat([h, s, p], dim=1)
        h = F.relu(self.enc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z, c, s, p):
        h = torch.cat([z, c, s, p], dim=1)
        h = F.relu(self.dec1(h))
        h = torch.cat([h, s, p], dim=1)
        h = F.relu(self.dec2(h))
        x_hat = self.out_x(h)
        return x_hat

    def predict_preparation(self, mu, c, s):
        # preparation은 조건에 강하게 묶이는 경우가 많으므로 c,s도 같이 주는 것이 일반적으로 유리
        inp = torch.cat([mu, c, s], dim=1)
        p_hat = self.out_p(inp)
        return p_hat

    def forward(self, x, c, s, p):
        mu, logvar = self.encoder(x, c, s, p)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c, s, p)
        p_hat = self.predict_preparation(mu, c, s)
        return x_hat, p_hat, mu, logvar
