"""PhaseB v2 — the CIN's neural core. Slim reimplementation of PolicyAlignedOutcomeNet
with the four audit-driven changes:

  1. TALENT CHANNEL (C2): talent_off/def_shrunk + talent_n_eff enter through their own
     embedding pathway so lever/context weights stop absorbing talent (deconfounding).
  2. Right-sized default: 4 blocks x 768 (ablate vs 8x1536 — configs/default.yaml).
  3. Learnable cross-task weights (Kendall-Gal): 10 heads with wildly different noise
     scales stop fighting; loss = sum_h [ NLL_h / (2*exp(s_h)) + s_h/2 ].
  4. Switch-weighted loss (C3): rows within `post_switch_games` of an observed context
     switch carry `switch_weight` (from the loader), upweighting exactly the regime
     the product queries.

Kept from v1: FiLM conditioning of the trunk on the lever vector; heteroscedastic
Gaussian heads (mu, log_sigma); PAR anti-leak penalty hook (pass probe_fn).
Internal sigma is used for RANKING uncertainty and training only — never for the band
a human sees (that is cin/conformal.py, by contract).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ["xGF", "xGA", "GF", "GA", "SF", "SA", "hits", "blocks", "takeaways", "giveaways"]


class FiLM(nn.Module):
    def __init__(self, lever_dim: int, width: int):
        super().__init__()
        self.g = nn.Linear(lever_dim, width)
        self.b = nn.Linear(lever_dim, width)

    def forward(self, h, levers):
        return h * (1 + torch.tanh(self.g(levers))) + self.b(levers)


class Block(nn.Module):
    def __init__(self, width: int, lever_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width, width)
        self.film = FiLM(lever_dim, width)
        self.drop = nn.Dropout(p_drop)

    def forward(self, h, levers):
        x = self.norm(h)
        a, g = self.fc1(x).chunk(2, dim=-1)      # SwiGLU
        x = self.fc2(F.silu(g) * a)
        x = self.film(x, levers)
        return h + self.drop(x)


class PhaseBv2(nn.Module):
    def __init__(self, ctx_dim: int, lever_dim: int = 17, usage_dim: int = 384,
                 talent_dim: int = 4, width: int = 768, depth: int = 4):
        super().__init__()
        self.ctx_in = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, width))
        self.usage_in = nn.Linear(usage_dim, width)
        # C2: talent gets its OWN pathway -> cannot be silently re-derived from levers
        self.talent_in = nn.Sequential(nn.Linear(talent_dim, 64), nn.SiLU(), nn.Linear(64, width))
        self.blocks = nn.ModuleList([Block(width, lever_dim) for _ in range(depth)])
        self.head_mu = nn.Linear(width, len(HEADS))
        self.head_logsig = nn.Linear(width, len(HEADS))
        self.task_logvar = nn.Parameter(torch.zeros(len(HEADS)))   # Kendall-Gal s_h

    def forward(self, ctx, levers, h_usage, talent):
        h = self.ctx_in(ctx) + self.usage_in(h_usage) + self.talent_in(talent)
        for blk in self.blocks:
            h = blk(h, levers)
        return self.head_mu(h), self.head_logsig(h).clamp(-6, 4)

    def loss(self, mu, logsig, y, exposure_w, switch_w, probe_fn=None, par_weight=0.1):
        """Heteroscedastic NLL, exposure- and switch-weighted, task-balanced."""
        nll = 0.5 * ((y - mu) ** 2 * torch.exp(-2 * logsig) + 2 * logsig)   # (B, H)
        w = (exposure_w * switch_w).unsqueeze(-1)                            # (B, 1)
        per_head = (nll * w).sum(0) / w.sum(0).clamp(min=1e-8)               # (H,)
        task_w = torch.exp(-self.task_logvar)
        total = (per_head * task_w + 0.5 * self.task_logvar).sum()
        if probe_fn is not None:                                             # PAR anti-leak
            total = total + par_weight * probe_fn(mu)
        return total, per_head.detach()

    @torch.no_grad()
    def predict_mu(self, batch) -> torch.Tensor:
        mu, _ = self.forward(*batch)
        return mu


def switch_weights(df, post_switch_games: int = 10, weight: float = 3.0):
    """C3: rows within N games after a team change for the focal player get upweighted.
    Expects df sorted by (playerId, date) with a `games_since_team_change` column
    (built in data/build_perfect_windows.py from teamId transitions)."""
    import numpy as np
    g = df["games_since_team_change"].values
    return torch.as_tensor(np.where(g <= post_switch_games, weight, 1.0), dtype=torch.float32)
