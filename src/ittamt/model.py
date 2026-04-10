from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_2d_sincos_pos_embed(dim: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D positional embedding expects dim divisible by 4, got {dim}")

    half = dim // 2
    quarter = dim // 4
    y = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1)
    omega = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter))

    y_emb = torch.cat([torch.sin(y * omega), torch.cos(y * omega)], dim=1)
    x_emb = torch.cat([torch.sin(x * omega), torch.cos(x * omega)], dim=1)
    y_emb = y_emb[:, None, :].expand(height, width, half)
    x_emb = x_emb[None, :, :].expand(height, width, half)
    pos = torch.cat([y_emb, x_emb], dim=-1)
    return pos.permute(2, 0, 1).unsqueeze(0)


class ConvStem(nn.Module):
    """Convolutional tokenizer that preserves 2D layout before flattening."""

    def __init__(self, in_ch: int = 1, dim: int = 256):
        super().__init__()
        mid1 = dim // 4
        mid2 = dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid1, 3, stride=2, padding=1),
            nn.GroupNorm(1, mid1),
            nn.GELU(),
            nn.Conv2d(mid1, mid2, 3, stride=2, padding=1),
            nn.GroupNorm(1, mid2),
            nn.GELU(),
            nn.Conv2d(mid2, dim, 3, stride=2, padding=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Local2DBlock(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.GroupNorm(1, dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(self.pw2(x))
        return residual + x


class SmartMoEFFN(nn.Module):
    """Top-k MoE with context-aware routing and load-balancing regularization."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.context_proj = nn.Linear(dim, dim)
        self.quality_proj = nn.Linear(1, dim)
        self.router = nn.Linear(dim, num_experts)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.noise_scale = nn.Parameter(torch.tensor(0.35))
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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_features = self.router_norm(x)
        if context is not None:
            router_features = router_features + self.context_proj(self.context_norm(context))
        if quality is not None:
            q = quality
            if q.ndim == 1:
                q = q[:, None]
            if q.ndim == 2:
                q = q[:, None, :]
            router_features = router_features + self.quality_proj(q)

        logits = self.router(router_features) / self.temperature.clamp(min=0.25)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_scale.clamp(min=0.0)

        topk_logits, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        gates = F.softmax(topk_logits, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[..., k]
            gate = gates[..., k].unsqueeze(-1).to(dtype=out.dtype)
            expert_out = torch.zeros_like(x)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    expert_out[mask] = self.experts[e](x[mask]).to(dtype=expert_out.dtype)
            out = out + gate * expert_out

        probs = F.softmax(logits, dim=-1)
        importance = probs.mean(dim=(0, 1))
        target = torch.full_like(importance, 1.0 / self.num_experts)
        load_balance = F.mse_loss(importance, target)
        router_z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
        aux = {
            "load_balance": load_balance,
            "router_z": router_z,
            "entropy": entropy,
            "aux": load_balance + 0.001 * router_z - 0.001 * entropy,
        }
        return self.dropout(out), aux


class GlobalMoEBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, num_experts: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SmartMoEFFN(dim, hidden_dim, num_experts=num_experts, top_k=top_k, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False, key_padding_mask=key_padding_mask)
        x = x + y
        y, aux = self.moe(self.norm2(x), context=context, quality=quality)
        x = x + y
        return x, aux


class SequenceEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float, num_experts: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GlobalMoEBlock(dim, heads, mlp_ratio, num_experts=num_experts, top_k=top_k, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux_total = {
            "load_balance": x.new_tensor(0.0),
            "router_z": x.new_tensor(0.0),
            "entropy": x.new_tensor(0.0),
            "aux": x.new_tensor(0.0),
        }
        for block in self.blocks:
            x, aux = block(x, context=context, quality=quality, key_padding_mask=key_padding_mask)
            for key in aux_total:
                aux_total[key] = aux_total[key] + aux[key]
        x = self.norm(x)
        return x, aux_total


class StructureHeads(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.textness = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
        )
        self.baseline = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
        )
        self.token_confidence = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, grid: torch.Tensor, tokens: torch.Tensor) -> "StructureOutputs":
        return StructureOutputs(
            textness_logits=self.textness(grid),
            baseline_logits=self.baseline(grid),
            token_confidence=self.token_confidence(tokens),
        )


class RefinementSourceEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int, heads: int, mlp_ratio: float, num_experts: int, top_k: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = SequenceEncoder(dim, depth, heads, mlp_ratio, num_experts, top_k, dropout=dropout)

    def forward(self, ids: torch.Tensor, quality: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask = ids.eq(1)
        x = self.token_emb(ids)
        x = self.dropout(x + self.pos_emb[:, : x.shape[1], :])
        x, aux = self.encoder(x, quality=quality, key_padding_mask=pad_mask)
        return x, aux["aux"]


class RefinementDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, num_experts: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.src_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.vis_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.moe = SmartMoEFFN(dim, hidden_dim, num_experts=num_experts, top_k=top_k, dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        src: torch.Tensor,
        vis: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        vis_key_padding_mask: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = self.norm1(q)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        q = q + y

        y = self.norm2(q)
        y, _ = self.src_attn(y, src, src, need_weights=False, key_padding_mask=src_key_padding_mask)
        q = q + y

        y = self.norm3(q)
        y, _ = self.vis_attn(y, vis, vis, need_weights=False, key_padding_mask=vis_key_padding_mask)
        q = q + y

        y, aux = self.moe(self.norm4(q), context=src, quality=quality)
        q = q + y
        return q, aux


class IterativeRefinementDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        num_experts: int,
        top_k: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.source_encoder = RefinementSourceEncoder(
            vocab_size=vocab_size,
            dim=dim,
            depth=max(1, depth // 2),
            heads=heads,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            top_k=top_k,
            max_len=max_len,
            dropout=dropout,
        )
        self.query_pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.query_quality = nn.Linear(1, dim)
        self.layers = nn.ModuleList(
            [
                RefinementDecoderLayer(dim, heads, mlp_ratio, num_experts=num_experts, top_k=top_k, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        source_ids: torch.Tensor,
        source_quality: torch.Tensor | None = None,
        visual_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_memory, source_aux = self.source_encoder(source_ids, quality=source_quality)
        batch = source_ids.shape[0]
        query = self.query_pos[:, : self.max_len, :].expand(batch, -1, -1)
        if source_quality is not None:
            q = source_quality
            if q.ndim == 1:
                q = q[:, None]
            query = query + self.query_quality(q[:, None, :])

        aux_total = {
            "load_balance": source_aux.new_tensor(0.0),
            "router_z": source_aux.new_tensor(0.0),
            "entropy": source_aux.new_tensor(0.0),
            "aux": source_aux.new_tensor(0.0),
        }
        for layer in self.layers:
            query, aux = layer(
                query,
                source_memory,
                visual_tokens,
                src_key_padding_mask=source_ids.eq(1),
                vis_key_padding_mask=visual_key_padding_mask,
                quality=source_quality,
            )
            for key in aux_total:
                aux_total[key] = aux_total[key] + aux[key]
        logits = self.lm_head(self.out_norm(query))
        aux_total["aux"] = aux_total["aux"] + source_aux
        return logits, aux_total


@dataclass
class StrideMoEConfig:
    vocab_size: int
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    num_experts: int = 8
    top_k: int = 2
    local_depth: int = 3
    refine_depth: int = 4
    refine_iters: int = 2
    max_refine_len: int = 130


@dataclass
class OCRMemory:
    grid: torch.Tensor
    tokens: torch.Tensor
    size: tuple[int, int]
    quality: torch.Tensor


@dataclass
class StructureOutputs:
    textness_logits: torch.Tensor
    baseline_logits: torch.Tensor
    token_confidence: torch.Tensor


@dataclass
class RefinementOutputs:
    logits: torch.Tensor
    pred_ids: torch.Tensor


@dataclass
class OCRForwardOutputs:
    coarse_logits: torch.Tensor
    coarse_pred_ids: torch.Tensor
    coarse_quality: torch.Tensor
    memory: OCRMemory
    structure: StructureOutputs
    refine_logits: torch.Tensor | None
    refine_aux: dict[str, torch.Tensor]
    coarse_aux: dict[str, torch.Tensor]


class StrideMoEOCR(nn.Module):
    """Dual-path OCR model with 2D tokenization, smarter MoE routing, and iterative refinement."""

    def __init__(self, cfg: StrideMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = ConvStem(1, cfg.dim)
        self.local_blocks = nn.ModuleList([Local2DBlock(cfg.dim) for _ in range(cfg.local_depth)])
        self.global_blocks = nn.ModuleList(
            [
                GlobalMoEBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.num_experts, cfg.top_k)
                for _ in range(cfg.depth)
            ]
        )
        self.fuse_gate = nn.Linear(cfg.dim * 3, cfg.dim)
        self.fuse_proj = nn.Linear(cfg.dim * 2, cfg.dim)
        self.post_local = Local2DBlock(cfg.dim)
        self.structure_heads = StructureHeads(cfg.dim)
        self.coarse_norm = nn.LayerNorm(cfg.dim)
        self.coarse_head = nn.Linear(cfg.dim, cfg.vocab_size)
        self.refiner = IterativeRefinementDecoder(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            depth=cfg.refine_depth,
            heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            max_len=cfg.max_refine_len,
        )

    def _encode(self, x: torch.Tensor) -> OCRMemory:
        grid = self.stem(x)
        grid = grid + _build_2d_sincos_pos_embed(grid.shape[1], grid.shape[2], grid.shape[3], grid.device)

        for block in self.local_blocks:
            grid = block(grid)
        local_grid = grid
        local_tokens = local_grid.flatten(2).transpose(1, 2)

        tokens = local_tokens
        coarse_aux = {
            "load_balance": x.new_tensor(0.0),
            "router_z": x.new_tensor(0.0),
            "entropy": x.new_tensor(0.0),
            "aux": x.new_tensor(0.0),
        }
        for block in self.global_blocks:
            tokens, aux = block(tokens, context=local_tokens)
            for key in coarse_aux:
                coarse_aux[key] = coarse_aux[key] + aux[key]

        gate = torch.sigmoid(self.fuse_gate(torch.cat([tokens, local_tokens, tokens - local_tokens], dim=-1)))
        fused_tokens = gate * tokens + (1.0 - gate) * local_tokens
        fused_tokens = self.fuse_proj(torch.cat([fused_tokens, local_tokens], dim=-1))
        fused_grid = fused_tokens.transpose(1, 2).reshape_as(local_grid)
        fused_grid = self.post_local(fused_grid)

        quality = self._coarse_quality_from_tokens(fused_tokens)
        self._last_coarse_aux = coarse_aux
        return OCRMemory(grid=fused_grid, tokens=fused_tokens, size=(fused_grid.shape[2], fused_grid.shape[3]), quality=quality)

    def _coarse_quality_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # Used both for routing and as a refinement hint.
        return torch.sigmoid(tokens.pow(2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1).neg() + 1.0)

    def _coarse_logits(self, memory: OCRMemory) -> torch.Tensor:
        return self.coarse_head(self.coarse_norm(memory.tokens))

    def coarse_only(self, x: torch.Tensor) -> torch.Tensor:
        memory = self._encode(x)
        return self._coarse_logits(memory)

    def refine(
        self,
        memory: OCRMemory,
        source_ids: torch.Tensor,
        source_quality: torch.Tensor | None = None,
    ) -> RefinementOutputs:
        logits, _ = self.refiner(memory.tokens, source_ids, source_quality=source_quality if source_quality is not None else memory.quality)
        pred_ids = torch.argmax(logits, dim=-1)
        return RefinementOutputs(logits=logits, pred_ids=pred_ids)

    def forward(
        self,
        x: torch.Tensor,
        refine_source_ids: torch.Tensor | None = None,
        source_quality: torch.Tensor | None = None,
    ) -> OCRForwardOutputs:
        memory = self._encode(x)
        coarse_logits = self._coarse_logits(memory)
        coarse_pred_ids = torch.argmax(coarse_logits, dim=-1)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        coarse_quality = 1.0 - (
            -(coarse_probs * coarse_probs.clamp_min(1e-9).log()).sum(dim=-1) / math.log(max(2, coarse_logits.shape[-1]))
        ).mean(dim=-1, keepdim=True)

        structure = self.structure_heads(memory.grid, memory.tokens)
        refine_logits = None
        refine_aux = {
            "load_balance": x.new_tensor(0.0),
            "router_z": x.new_tensor(0.0),
            "entropy": x.new_tensor(0.0),
            "aux": x.new_tensor(0.0),
        }
        if refine_source_ids is not None:
            refine_logits, refine_aux = self.refiner(
                memory.tokens,
                refine_source_ids,
                source_quality=source_quality if source_quality is not None else coarse_quality,
            )

        coarse_aux = getattr(self, "_last_coarse_aux", {
            "load_balance": x.new_tensor(0.0),
            "router_z": x.new_tensor(0.0),
            "entropy": x.new_tensor(0.0),
            "aux": x.new_tensor(0.0),
        })
        return OCRForwardOutputs(
            coarse_logits=coarse_logits,
            coarse_pred_ids=coarse_pred_ids,
            coarse_quality=coarse_quality,
            memory=memory,
            structure=structure,
            refine_logits=refine_logits,
            refine_aux=refine_aux,
            coarse_aux=coarse_aux,
        )
