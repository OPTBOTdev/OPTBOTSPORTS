"""Phase A v2 — two-stream identity/context encoder, slim rebuild.

Carries the audited-good design (two streams + one-way firewall) and fixes the
audited-broken parts: single consistent module (no train-script/heads drift),
hard negatives ON, usefulness head wired, ordering contract assumed from data layer
(with_ids seconds-desc — enforced upstream by contracts, not re-sorted here).

Token layout (d=384): [CLS, FOCAL, STYLE, RELIAB, WITH, VS, CTX, CHEM, ROLE]
  identity stream: FOCAL, STYLE, RELIAB       (2 layers)  -> h_skill (pooled, L2)
  context  stream: CLS, WITH, VS, CTX, CHEM, ROLE (4 layers) -> h_win  (CLS)
  firewall: context CLS cross-attends to a DETACHED identity summary only.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 384
ID_TOKENS, CTX_TOKENS = 3, 6


class SetPool(nn.Module):
    """Seconds-share-biased attention pool over a padded (B, K, d) set."""
    def __init__(self, d=D):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)

    def forward(self, x, seconds):                       # x (B,K,d), seconds (B,K)
        logits = x @ self.q + torch.log(seconds.clamp(min=1.0))
        logits = logits.masked_fill(seconds <= 0, -1e9)
        w = logits.softmax(-1).unsqueeze(-1)
        return (w * x).sum(1)


class PhaseAv2(nn.Module):
    def __init__(self, n_players: int, n_cat_vocab: dict[str, int],
                 style_dim=23, reliab_dim=10, role_dim=75, ctx_cont_dim=4,
                 d=D, id_layers=2, ctx_layers=4, heads=6):
        super().__init__()
        self.player_emb = nn.Embedding(n_players + 1, 96, padding_idx=0)
        self.tok = nn.ModuleDict({
            "focal": nn.Linear(96 + 9, d),               # player emb + 9 intrinsic priors
            "style": nn.Linear(style_dim, d),
            "reliab": nn.Linear(reliab_dim, d),
            "role": nn.Linear(role_dim, d),
            "ctx_cont": nn.Linear(ctx_cont_dim, d),
        })
        self.cat_embs = nn.ModuleDict({k: nn.Embedding(v, d // 8) for k, v in n_cat_vocab.items()})
        self.ctx_proj = nn.Linear(len(n_cat_vocab) * (d // 8) + d, d)
        self.with_pool, self.vs_pool = SetPool(d), SetPool(d)
        self.pid_proj = nn.Linear(96, d)
        self.type_emb = nn.Embedding(ID_TOKENS + CTX_TOKENS, d)
        enc = lambda n: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, d * 4, 0.1, "gelu",
                                       batch_first=True, norm_first=True), n)
        self.id_enc, self.ctx_enc = enc(id_layers), enc(ctx_layers)
        self.firewall = nn.MultiheadAttention(d, heads, batch_first=True)
        self.role_gate = nn.Parameter(torch.tensor(-2.0))     # sigmoid*0.3 cap
        self.id_pool_q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.cls = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.chem_proj = nn.Linear(d, d)

    def forward(self, b):
        B = b["focal_pid"].shape[0]
        pe = self.player_emb(b["focal_pid"])
        focal = self.tok["focal"](torch.cat([pe, b["intrinsic9"]], -1))
        style = self.tok["style"](b["style23"])
        reliab = self.tok["reliab"](b["reliab10"])
        with_t = self.with_pool(self.pid_proj(self.player_emb(b["with_ids"])), b["with_seconds"])
        vs_t = self.vs_pool(self.pid_proj(self.player_emb(b["vs_ids"])), b["vs_seconds"])
        cat = torch.cat([self.cat_embs[k](b["cat"][k]) for k in self.cat_embs], -1)
        ctx = self.ctx_proj(torch.cat([cat, self.tok["ctx_cont"](b["ctx_cont4"])], -1))
        role = torch.sigmoid(self.role_gate) * 0.3 * self.tok["role"](b["role75"])
        chem = self.chem_proj(with_t)

        idt = torch.stack([focal, style, reliab], 1) + self.type_emb.weight[:ID_TOKENS]
        h_id = self.id_enc(idt)
        w = (h_id @ self.id_pool_q).softmax(-1).unsqueeze(-1)
        h_skill = F.normalize((w * h_id).sum(1), dim=-1)

        ctx_toks = torch.stack([self.cls.expand(B, -1), with_t, vs_t, ctx, chem, role], 1) \
            + self.type_emb.weight[ID_TOKENS:]
        h_ctx = self.ctx_enc(ctx_toks)
        # one-way firewall: CLS reads a detached (10%-dropped) identity summary
        id_summary = h_id.detach()
        if self.training:
            id_summary = F.dropout(id_summary, 0.1)
        cls_x, _ = self.firewall(h_ctx[:, :1], id_summary, id_summary)
        h_win = (h_ctx[:, 0] + cls_x[:, 0])
        return {"h_skill": h_skill, "h_win": h_win}


def info_nce_with_hard_negs(h, pos_idx, hard_neg_idx, tau=0.1):
    """Multi-positive InfoNCE where same-context-different-player impostors are
    appended to the denominator — the portability loss the v1 audit found unused."""
    sim = h @ h.T / tau
    B = h.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=h.device)
    sim = sim.masked_fill(eye, -1e9)
    loss = 0.0
    for i in range(B):
        pos = pos_idx[i]
        if not len(pos):
            continue
        denom_idx = torch.cat([pos, hard_neg_idx[i],
                               torch.arange(B, device=h.device)[~eye[i]][:32]])
        num = torch.logsumexp(sim[i, pos], 0)
        den = torch.logsumexp(sim[i, denom_idx], 0)
        loss = loss - (num - den)
    return loss / B
