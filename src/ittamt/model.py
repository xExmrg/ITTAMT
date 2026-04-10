from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """Fast local visual feature extractor."""

    def __init__(self, in_ch: int = 1, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim // 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoEFFN(nn.Module):
    """Top-k MoE feed-forward layer."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        router_logits = self.router(x)
        topk_logits, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        gates = F.softmax(topk_logits, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[..., k]  # [B, T]
            gate = gates[..., k].unsqueeze(-1).to(dtype=out.dtype)
            expert_out = torch.zeros_like(x)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    vals = self.experts[e](x[mask]).to(dtype=expert_out.dtype)
                    expert_out[mask] = vals
            out = out + gate * expert_out

        # Load-balance auxiliary loss (simple variant)
        probs = F.softmax(router_logits, dim=-1)
        importance = probs.mean(dim=(0, 1))
        target = torch.full_like(importance, 1.0 / self.num_experts)
        aux_loss = F.mse_loss(importance, target)
        return self.dropout(out), aux_loss


class MoETransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, num_experts: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SparseMoEFFN(dim, hidden_dim, num_experts=num_experts, top_k=top_k, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y, aux = self.moe(self.norm2(x))
        x = x + y
        return x, aux


@dataclass
class StrideMoEConfig:
    vocab_size: int
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    num_experts: int = 8
    top_k: int = 2


class StrideMoEOCR(nn.Module):
    """Lightweight OCR architecture with MoE transformer encoder + CTC head."""

    def __init__(self, cfg: StrideMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = ConvStem(1, cfg.dim)
        self.pos_emb = None
        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    mlp_ratio=cfg.mlp_ratio,
                    num_experts=cfg.num_experts,
                    top_k=cfg.top_k,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size)

    def _build_pos_emb(self, t: int, c: int, device: torch.device) -> torch.Tensor:
        if self.pos_emb is not None and self.pos_emb.shape[1] == t:
            return self.pos_emb
        pos = torch.arange(t, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2, device=device).float() * (-math.log(10000.0) / c))
        pe = torch.zeros(t, c, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pos_emb = pe.unsqueeze(0)
        return self.pos_emb

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, H, W]
        feats = self.stem(x)  # [B, C, H', W']
        feats = feats.mean(dim=2)  # collapse height -> [B, C, W']
        tokens = feats.transpose(1, 2)  # [B, T, C]
        pos = self._build_pos_emb(tokens.shape[1], tokens.shape[2], tokens.device)
        x = tokens + pos

        aux_total = x.new_tensor(0.0)
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux

        x = self.norm(x)
        logits = self.head(x)  # [B, T, V]
        return logits, aux_total / len(self.blocks)
