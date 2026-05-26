"""Student world model — residual MLP with LayerNorm + GELU."""
from __future__ import annotations
import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.ln(x)
        return x + self.fc2(self.drop(self.act(self.fc1(h))))


class StudentWorldModel(nn.Module):
    def __init__(self, obs_dim=4, act_dim=1, hidden_dim=256, num_layers=4,
                 use_gru=False, delta_limit=3.0, expansion=2, dropout=0.1):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.in_proj = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim, expansion, dropout) for _ in range(int(num_layers))]
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, obs_dim)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1.0e-3)

    def initial_hidden(self, batch_size, device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm, act_norm, hidden=None):
        h = self.in_proj(torch.cat([obs_norm, act_norm], dim=-1))
        for block in self.blocks:
            h = block(h)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(h.shape[0], h.device)
            hidden = self.gru(h, hidden)
            h = hidden
        raw = self.head(self.out_ln(h))
        delta = self.delta_limit * torch.tanh(raw / self.delta_limit)
        return delta, hidden