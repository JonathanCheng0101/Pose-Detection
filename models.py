import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """
    A  residual MLP block:
     - two Linear→BatchNorm→LeakyReLU→Dropout layers
     - adds input back to output (with padding if needed)
    """
    def __init__(self, in_dim=5, hidden=256, out_dim=10, p_drop=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(p_drop)
        )
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        residual = x
        x = self.block(x)
        if residual.shape[1] != x.shape[1]:
            residual = F.pad(residual, (0, x.shape[1] - residual.shape[1]))
        return self.out(x + residual)

class ResMLP2(nn.Module):
    """
    Two sequential residual blocks:
     - projects input if its dim ≠ hidden size
     - adds skip-connections at each block
    """
    def __init__(self, in_dim=5, h=256, out_dim=10, p_drop=0.4):
        super().__init__()
        self.proj = nn.Identity() if in_dim == h else nn.Linear(in_dim, h)
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, h), nn.BatchNorm1d(h),
            nn.LeakyReLU(0.1), nn.Dropout(p_drop),
            nn.Linear(h, h),       nn.BatchNorm1d(h),
            nn.LeakyReLU(0.1), nn.Dropout(p_drop)
        )
        self.block2 = nn.Sequential(
            nn.Linear(h, h),       nn.BatchNorm1d(h),
            nn.LeakyReLU(0.1), nn.Dropout(p_drop),
            nn.Linear(h, h),       nn.BatchNorm1d(h),
            nn.LeakyReLU(0.1), nn.Dropout(p_drop)
        )
        self.out = nn.Linear(h, out_dim)

    def forward(self, x):
        h1 = self.block1(x) + self.proj(x)
        h2 = self.block2(h1) + h1
        return self.out(h2)
