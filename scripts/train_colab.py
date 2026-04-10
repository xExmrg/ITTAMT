#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import textwrap
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow script execution without editable install (e.g., raw Colab clone)."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from ittamt.data import DataConfig, build_dataloaders
from ittamt.model import StrideMoEConfig, StrideMoEOCR
from ittamt.tokenizer import CharTokenizer


def ctc_loss_from_logits(logits, labels, label_lengths, blank_id: int):
    # logits [B, T, V] -> log_probs [T, B, V]
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    bsz, t, _ = logits.shape
    input_lengths = torch.full((bsz,), t, dtype=torch.long, device=logits.device)

    flat_targets = []
    for i in range(bsz):
        length = int(label_lengths[i].item())
        flat_targets.append(labels[i, :length])
    targets = torch.cat(flat_targets) if flat_targets else torch.tensor([], device=logits.device, dtype=torch.long)

    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        label_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def _as_output_dict(output) -> dict:
    """Normalize a variety of model outputs into a dict."""
    if isinstance(output, dict):
        return output
    if torch.is_tensor(output):
        return {"logits": output}
    if isinstance(output, tuple) and len(output) == 2 and torch.is_tensor(output[0]):
        return {"logits": output[0], "aux_loss": output[1]}
    if isinstance(output, tuple) and len(output) == 3 and torch.is_tensor(output[0]):
        return {"logits": output[0], "aux_loss": output[1], "extra": output[2]}
    return {"raw": output}


def _extract_logits_pair(outputs: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (coarse_logits, refined_logits). Either can be None."""
    for key in ("logits_pair", "logits_list", "logits_all"):
        val = outputs.get(key)
        if isinstance(val, (list, tuple)) and val and all(torch.is_tensor(x) for x in val):
            coarse = val[0]
            refined = val[-1] if len(val) > 1 else None
            return coarse, refined

    coarse = (
        outputs.get("logits_coarse")
        or outputs.get("coarse_logits")
        or outputs.get("ctc_logits")
        or outputs.get("logits")
    )
    refined = (
        outputs.get("logits_refined")
        or outputs.get("refined_logits")
        or outputs.get("logits_final")
        or outputs.get("final_logits")
    )
    coarse = coarse if torch.is_tensor(coarse) else None
    refined = refined if torch.is_tensor(refined) else None
    return coarse, refined


def _extract_seq_logits(outputs: dict) -> torch.Tensor | None:
    for key in ("seq_logits", "refine_seq_logits", "refined_seq_logits", "decoder_logits"):
        val = outputs.get(key)
        if torch.is_tensor(val):
            return val
    struct = outputs.get("refine")
    if isinstance(struct, dict):
        val = struct.get("seq_logits")
        if torch.is_tensor(val):
            return val
    return None


def _extract_aux_losses(outputs: dict) -> torch.Tensor:
    aux_total = None
    for key in ("aux_loss", "moe_aux_loss", "router_aux_loss", "load_balance_loss", "lb_loss"):
        val = outputs.get(key)
        if torch.is_tensor(val):
            aux_total = val if aux_total is None else (aux_total + val)
    val = outputs.get("aux_losses")
    if isinstance(val, dict):
        for v in val.values():
            if torch.is_tensor(v):
                aux_total = v if aux_total is None else (aux_total + v)
    elif isinstance(val, (list, tuple)):
        for v in val:
            if torch.is_tensor(v):
                aux_total = v if aux_total is None else (aux_total + v)
    if aux_total is None:
        return torch.tensor(0.0)
    return aux_total


def _forward_model(model, images: torch.Tensor, batch: dict, tokenizer: CharTokenizer, refine_steps: int | None):
    """Call model with optional supervision kwargs if its forward() supports them."""
    kwargs = {}
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
    except Exception:
        params = {}

    def _maybe(name: str, value):
        if name in params:
            kwargs[name] = value

    # Common names used by research-y models.
    _maybe("labels", batch.get("labels"))
    _maybe("label_lengths", batch.get("label_lengths"))
    _maybe("seq_labels", batch.get("seq_labels"))
    _maybe("seq_label_lengths", batch.get("seq_label_lengths"))
    _maybe("struct_targets", batch.get("struct"))
    _maybe("struct", batch.get("struct"))
    _maybe("tokenizer", tokenizer)
    if refine_steps is not None:
        _maybe("refine_steps", refine_steps)
        _maybe("num_refine_steps", refine_steps)

    return model(images, **kwargs)


def _to_device_struct(struct: object, device: torch.device) -> object:
    if not isinstance(struct, dict):
        return struct
    out: dict = {}
    for k, v in struct.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _structure_losses_from_outputs(outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute dense structure losses when available."""
    struct_tgt = batch.get("struct")
    if not isinstance(struct_tgt, dict):
        return torch.tensor(0.0), {}

    valid = struct_tgt.get("valid")
    if not torch.is_tensor(valid):
        return torch.tensor(0.0), {}
    pred_device = None
    for v in outputs.values():
        if torch.is_tensor(v):
            pred_device = v.device
            break
    struct_pred_device = None
    if pred_device is None and isinstance(outputs.get("struct"), dict):
        for v in outputs["struct"].values():
            if torch.is_tensor(v):
                struct_pred_device = v.device
                break
    device = pred_device or struct_pred_device or valid.device
    valid_f = valid.to(dtype=torch.float32, device=device)
    denom = valid_f.sum().clamp(min=1.0)

    # Find predicted struct dict or flat keys.
    struct_pred = outputs.get("struct")
    if not isinstance(struct_pred, dict):
        struct_pred = outputs

    losses: dict[str, torch.Tensor] = {}
    stats: dict[str, float] = {}

    def _masked_mean(loss_map: torch.Tensor) -> torch.Tensor:
        # Reduce per-sample then mask.
        per = loss_map.view(loss_map.shape[0], -1).mean(dim=1)
        return (per * valid_f).sum() / denom

    # Heatmaps: BCEWithLogits vs [0,1] targets.
    for name, pred_keys, tgt_key in (
        ("text_mask", ("text_mask_logits", "mask_logits", "text_mask_logit"), "text_mask"),
        ("baseline", ("baseline_heatmap_logits", "baseline_logits", "baseline_logit"), "baseline_heatmap"),
        ("char_centers", ("char_center_heatmap_logits", "char_heatmap_logits", "centers_logits"), "char_center_heatmap"),
    ):
        pred = None
        for k in pred_keys:
            v = struct_pred.get(k)
            if torch.is_tensor(v):
                pred = v
                break
        if pred is None:
            continue
        tgt = struct_tgt.get(tgt_key)
        if not torch.is_tensor(tgt):
            continue
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if tgt.ndim == 3:
            tgt = tgt.unsqueeze(1)
        if pred.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(tgt.to(dtype=torch.float32), size=pred.shape[-2:], mode="bilinear", align_corners=False)
        loss_map = F.binary_cross_entropy_with_logits(pred, tgt.to(device=pred.device, dtype=pred.dtype), reduction="none")
        losses[name] = _masked_mean(loss_map)

    # Density: L1 regression.
    density_pred = None
    for k in ("density_logits", "density_pred", "density"):
        v = struct_pred.get(k)
        if torch.is_tensor(v):
            density_pred = v
            break
    if density_pred is not None:
        tgt = struct_tgt.get("density")
        if torch.is_tensor(tgt):
            if density_pred.ndim == 3 and density_pred.shape[1] == 1:
                density_pred = density_pred.squeeze(1)
            if density_pred.ndim == 2 and tgt.ndim == 2 and density_pred.shape[1] != tgt.shape[1]:
                tgt_1d = tgt.unsqueeze(1)
                tgt_1d = F.interpolate(tgt_1d, size=density_pred.shape[1], mode="linear", align_corners=False).squeeze(1)
            else:
                tgt_1d = tgt
            l1 = F.l1_loss(density_pred.to(dtype=torch.float32), tgt_1d.to(device=density_pred.device, dtype=torch.float32), reduction="none")
            losses["density"] = (l1.mean(dim=1) * valid_f).sum() / denom

    if not losses:
        return torch.tensor(0.0), {}

    total = sum(losses.values())
    for k, v in losses.items():
        stats[f"struct/{k}"] = float(v.detach().cpu().item())
    return total, stats


def greedy_decode(logits: torch.Tensor, tokenizer: CharTokenizer) -> list[str]:
    ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [tokenizer.decode_ctc(row) for row in ids]


def pack_sequence_batch(tokenizer: CharTokenizer, texts: list[str], device: torch.device, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded, lengths = tokenizer.batch_encode_sequence(texts, max_length=max_length)
    padded_len = max([len(seq) for seq in encoded] + [1])
    padded = torch.full((len(encoded), padded_len), tokenizer.pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(encoded):
        if seq:
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


def sequence_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id)


def pad_sequence_targets(targets: torch.Tensor, pad_id: int, length: int) -> torch.Tensor:
    if targets.shape[1] == length:
        return targets
    padded = torch.full((targets.shape[0], length), pad_id, dtype=targets.dtype, device=targets.device)
    copy_len = min(length, targets.shape[1])
    if copy_len > 0:
        padded[:, :copy_len] = targets[:, :copy_len]
    return padded


def _structure_targets(
    struct: dict[str, torch.Tensor],
    target_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    text_mask = struct["text_mask"].to(device)
    baseline = struct["baseline_heatmap"].to(device)
    centers = struct["char_center_heatmap"].to(device)
    density = struct["density"].to(device)
    valid = struct["valid"].to(device).float()

    text_mask = F.interpolate(text_mask, size=target_hw, mode="bilinear", align_corners=False)
    baseline = F.interpolate(baseline, size=target_hw, mode="bilinear", align_corners=False)
    centers = F.interpolate(centers, size=target_hw, mode="bilinear", align_corners=False)
    density = F.interpolate(density.unsqueeze(1), size=target_hw[1], mode="linear", align_corners=False).squeeze(1)
    return text_mask, baseline, centers, valid, density


def structure_loss(batch_struct: dict[str, torch.Tensor], outputs, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    if int(batch_struct["valid"].sum().item()) <= 0:
        zero = outputs.textness_logits.new_tensor(0.0)
        return zero, {"text_mask": 0.0, "baseline": 0.0, "token_conf": 0.0}

    target_hw = outputs.textness_logits.shape[-2:]
    text_mask, baseline, centers, valid, density = _structure_targets(batch_struct, target_hw, device)
    valid_4d = valid[:, None, None, None]
    valid_1d = valid[:, None]

    text_loss_raw = F.binary_cross_entropy_with_logits(outputs.textness_logits, text_mask, reduction="none")
    baseline_loss_raw = F.binary_cross_entropy_with_logits(outputs.baseline_logits, baseline, reduction="none")
    pixel_count = valid_4d.sum().clamp_min(1.0) * text_loss_raw.shape[-1] * text_loss_raw.shape[-2]
    text_loss = (text_loss_raw * valid_4d).sum() / pixel_count
    baseline_loss = (baseline_loss_raw * valid_4d).sum() / pixel_count

    token_target = F.interpolate(density.unsqueeze(1), size=outputs.token_confidence.shape[1], mode="linear", align_corners=False).squeeze(1)
    token_loss_raw = F.binary_cross_entropy_with_logits(outputs.token_confidence.squeeze(-1), token_target, reduction="none")
    token_loss = (token_loss_raw * valid_1d).sum() / (valid_1d.sum().clamp_min(1.0) * token_loss_raw.shape[-1])

    total = text_loss + baseline_loss + 0.5 * token_loss
    return total, {"text_mask": float(text_loss.item()), "baseline": float(baseline_loss.item()), "token_conf": float(token_loss.item())}


def decode_ids_to_texts(tokenizer: CharTokenizer, ids: torch.Tensor, mode: str) -> list[str]:
    rows = ids.detach().cpu().tolist()
    if mode == "ctc":
        return [tokenizer.decode_ctc(row) for row in rows]
    return [tokenizer.decode_sequence(row) for row in rows]


def edit_distance(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 1.0 if pred else 0.0
    return edit_distance(pred, ref) / max(1, len(ref))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def running_in_colab() -> bool:
    return "COLAB_GPU" in os.environ or Path("/content").exists()


def default_dataset_cache_dir() -> str:
    if running_in_colab():
        return "/content/ittamt_datasets"
    return str(Path.home() / ".cache" / "ittamt_datasets")


def default_output_dir() -> str:
    if running_in_colab():
        return "/content/ittamt_artifacts/stride_moe"
    return "artifacts/stride_moe"


def resolve_batch_size(requested_batch_size: int, device: torch.device) -> tuple[int, str]:
    if requested_batch_size > 0:
        return requested_batch_size, "manual"
    if device.type != "cuda":
        return 16, "cpu-safe default"

    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_gb >= 120:
        return 256, "auto(>=120GB VRAM)"
    if total_gb >= 90:
        return 192, "auto(>=90GB VRAM)"
    if total_gb >= 80:
        return 160, "auto(>=80GB VRAM)"
    if total_gb >= 48:
        return 96, "auto(>=48GB VRAM)"
    if total_gb >= 24:
        return 48, "auto(>=24GB VRAM)"
    return 16, "auto(<24GB VRAM)"


def configure_cuda_runtime(device: torch.device) -> dict[str, str | float]:
    runtime: dict[str, str | float] = {"device": str(device)}
    if device.type != "cuda":
        return runtime

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    runtime.update(
        {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_gb": round(total_bytes / (1024**3), 1),
            "gpu_free_gb": round(free_bytes / (1024**3), 1),
        }
    )
    return runtime


def _format_preview_text(label: str, value: str, width: int = 60) -> list[str]:
    normalized = value.replace("\n", " | ")
    wrapped = textwrap.wrap(f"{label}: {normalized}", width=width) or [f"{label}: "]
    return wrapped[:4]


def save_eval_preview(
    preview_images: list[Image.Image],
    refs: list[str],
    preds: list[str],
    out_path: Path,
    image_width: int,
    max_items: int,
) -> None:
    if not preview_images:
        return

    preview_dir = out_path.parent
    preview_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    card_width = max(560, image_width + 48)
    image_box_height = 128
    text_box_height = 84
    cards: list[Image.Image] = []

    for idx, (image, ref, pred) in enumerate(zip(preview_images[:max_items], refs[:max_items], preds[:max_items]), start=1):
        canvas = Image.new("RGB", (card_width, image_box_height + text_box_height), "white")
        draw = ImageDraw.Draw(canvas)

        bordered = ImageOps.expand(image.convert("RGB"), border=2, fill="black")
        fitted = ImageOps.contain(bordered, (card_width - 20, image_box_height - 12))
        canvas.paste(fitted, ((card_width - fitted.width) // 2, 6))

        text_y = image_box_height + 2
        draw.text((8, text_y), f"sample {idx}", fill="black", font=font)
        text_y += 14
        for line in _format_preview_text("ref", ref):
            draw.text((8, text_y), line, fill="black", font=font)
            text_y += 12
        for line in _format_preview_text("pred", pred):
            draw.text((8, text_y), line, fill="black", font=font)
            text_y += 12
        cards.append(canvas)

    preview = Image.new("RGB", (card_width, len(cards) * cards[0].height), "white")
    for row, card in enumerate(cards):
        preview.paste(card, (0, row * card.height))
    preview.save(out_path)


def _print_dataset_summary(summary: dict[str, dict[str, int] | list[str]]) -> None:
    print("dataset mix:")
    for split_name in ["train", "validation"]:
        split_summary = summary.get(split_name, {})
        total = sum(split_summary.values()) if isinstance(split_summary, dict) else 0
        print(f"  {split_name}: total={total}")
        if isinstance(split_summary, dict):
            for dataset_name, count in split_summary.items():
                print(f"    - {dataset_name}: {count}")
    warnings = summary.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            print(f"  warning: {warning}")


def main():
    ap = argparse.ArgumentParser(description="Train STRIDE-MoE OCR on Colab")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto-scale from detected GPU VRAM")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--refine-steps", type=int, default=2, help="Iterative refinement steps (if supported by model)")
    ap.add_argument("--ctc-coarse-weight", type=float, default=1.0)
    ap.add_argument("--ctc-refine-weight", type=float, default=1.0)
    ap.add_argument("--seq-refine-weight", type=float, default=0.0, help="Cross-entropy loss on seq logits (if model returns them)")
    ap.add_argument("--struct-weight", type=float, default=0.2, help="Weight of structure supervision losses (synthetic-only)")
    ap.add_argument("--aux-weight", type=float, default=0.05, help="Weight for routing/load-balance aux losses")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="0 disables gradient clipping")
    ap.add_argument("--image-height", type=int, default=64)
    ap.add_argument("--image-width", type=int, default=512)
    ap.add_argument("--synthetic-samples", type=int, default=40000)
    ap.add_argument("--synthetic-val-samples", type=int, default=3000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--preview-count", type=int, default=4)
    ap.add_argument("--dataset-cache-dir", type=str, default=None)
    ap.add_argument("--allow-non-cuda", action="store_true", help="Allow fallback to CPU/MPS instead of requiring CUDA")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    device = get_device()
    if device.type != "cuda" and not args.allow_non_cuda:
        raise RuntimeError(
            "train_colab.py requires a CUDA GPU by default. "
            "In Colab, switch Runtime -> Change runtime type -> GPU. "
            "For local debugging only, pass --allow-non-cuda."
        )

    runtime_info = configure_cuda_runtime(device)
    batch_size, batch_size_mode = resolve_batch_size(args.batch_size, device)
    dataset_cache_dir = Path(args.dataset_cache_dir or default_dataset_cache_dir()).resolve()
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("runtime config:")
    print(f"  device: {runtime_info['device']}")
    if device.type == "cuda":
        print(f"  gpu: {runtime_info['gpu_name']}")
        print(f"  gpu_vram_total_gb: {runtime_info['gpu_total_gb']}")
        print(f"  gpu_vram_free_gb: {runtime_info['gpu_free_gb']}")
    print(f"  batch_size: {batch_size} ({batch_size_mode})")
    print(f"  num_workers: {args.num_workers}")
    print(f"  prefetch_factor: {args.prefetch_factor}")
    print(f"  dataset_cache_dir: {dataset_cache_dir}")
    print(f"  output_dir: {output_dir}")

    tokenizer = CharTokenizer.build_default()
    tokenizer.save(str(output_dir / "tokenizer.json"))

    data_cfg = DataConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=device.type == "cuda",
        dataset_cache_dir=str(dataset_cache_dir),
        synthetic_samples=args.synthetic_samples,
        synthetic_val_samples=args.synthetic_val_samples,
    )
    train_loader, val_loader, dataset_summary = build_dataloaders(tokenizer, data_cfg)
    _print_dataset_summary(dataset_summary)

    model_cfg = StrideMoEConfig(
        vocab_size=len(tokenizer.itos),
        dim=256,
        depth=6,
        heads=8,
        num_experts=8,
        top_k=2,
        local_depth=3,
        refine_depth=4,
        refine_iters=2,
        max_refine_len=data_cfg.max_label_len + 2,
    )
    model = StrideMoEOCR(model_cfg)
    if device.type == "cuda":
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    preview_dir = output_dir / "eval_previews"
    autocast_kwargs = {
        "device_type": "cuda" if device.type == "cuda" else "cpu",
        "dtype": torch.bfloat16,
        "enabled": device.type == "cuda",
    }

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = batch["labels"].to(device, non_blocking=True)
            label_lengths = batch["label_lengths"].to(device, non_blocking=True)
            seq_labels = pad_sequence_targets(
                batch["seq_labels"].to(device, non_blocking=True),
                tokenizer.pad_id,
                model_cfg.max_refine_len,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**autocast_kwargs):
                outputs = model(images)
                ctc = ctc_loss_from_logits(outputs.coarse_logits, labels, label_lengths, tokenizer.blank_id)

                coarse_texts = decode_ids_to_texts(tokenizer, outputs.coarse_pred_ids, "ctc")
                refine_source_ids, _ = pack_sequence_batch(tokenizer, coarse_texts, device, model_cfg.max_refine_len)
                refine_logits_1, refine_aux_1 = model.refiner(
                    outputs.memory.tokens,
                    refine_source_ids,
                    source_quality=outputs.coarse_quality,
                )
                refine_loss_1 = sequence_ce_loss(refine_logits_1, seq_labels, tokenizer.pad_id)

                refine_loss = refine_loss_1
                refine_aux = refine_aux_1
                final_refine_logits = refine_logits_1
                if model_cfg.refine_iters > 1:
                    refine_texts = decode_ids_to_texts(tokenizer, refine_logits_1.argmax(dim=-1), "sequence")
                    refine_source_ids_2, _ = pack_sequence_batch(tokenizer, refine_texts, device, model_cfg.max_refine_len)
                    refine_logits_2, refine_aux_2 = model.refiner(
                        outputs.memory.tokens,
                        refine_source_ids_2,
                        source_quality=outputs.coarse_quality,
                    )
                    refine_loss_2 = sequence_ce_loss(refine_logits_2, seq_labels, tokenizer.pad_id)
                    refine_loss = 0.5 * refine_loss_1 + refine_loss_2
                    refine_aux = {key: refine_aux_1[key] + refine_aux_2[key] for key in refine_aux_1}
                    final_refine_logits = refine_logits_2

                struct_loss, _ = structure_loss(batch["struct"], outputs.structure, device)
                moe_aux = outputs.coarse_aux["aux"] + refine_aux["aux"]
                loss = ctc + 0.75 * refine_loss + 0.15 * struct_loss + 0.02 * moe_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", ctc=f"{ctc.item():.4f}", refine=f"{refine_loss.item():.4f}")

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        coarse_cers = []
        refine_cers = []
        preview_payload: tuple[list[Image.Image], list[str], list[str]] | None = None
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                images = batch["image"].to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = batch["labels"].to(device, non_blocking=True)
                label_lengths = batch["label_lengths"].to(device, non_blocking=True)
                seq_labels = pad_sequence_targets(
                    batch["seq_labels"].to(device, non_blocking=True),
                    tokenizer.pad_id,
                    model_cfg.max_refine_len,
                )

                with torch.autocast(**autocast_kwargs):
                    outputs = model(images)
                    ctc = ctc_loss_from_logits(outputs.coarse_logits, labels, label_lengths, tokenizer.blank_id)
                    coarse_texts = decode_ids_to_texts(tokenizer, outputs.coarse_pred_ids, "ctc")
                    refine_source_ids, _ = pack_sequence_batch(tokenizer, coarse_texts, device, model_cfg.max_refine_len)
                    refine_logits_1, refine_aux_1 = model.refiner(
                        outputs.memory.tokens,
                        refine_source_ids,
                        source_quality=outputs.coarse_quality,
                    )
                    refine_loss_1 = sequence_ce_loss(refine_logits_1, seq_labels, tokenizer.pad_id)
                    refine_texts = decode_ids_to_texts(tokenizer, refine_logits_1.argmax(dim=-1), "sequence")

                    refine_loss = refine_loss_1
                    refine_aux = refine_aux_1
                    final_refine_logits = refine_logits_1
                    if model_cfg.refine_iters > 1:
                        refine_source_ids_2, _ = pack_sequence_batch(tokenizer, refine_texts, device, model_cfg.max_refine_len)
                        refine_logits_2, refine_aux_2 = model.refiner(
                            outputs.memory.tokens,
                            refine_source_ids_2,
                            source_quality=outputs.coarse_quality,
                        )
                        refine_loss_2 = sequence_ce_loss(refine_logits_2, seq_labels, tokenizer.pad_id)
                        refine_loss = 0.5 * refine_loss_1 + refine_loss_2
                        refine_aux = {key: refine_aux_1[key] + refine_aux_2[key] for key in refine_aux_1}
                        final_refine_logits = refine_logits_2

                    struct_loss, _ = structure_loss(batch["struct"], outputs.structure, device)
                    moe_aux = outputs.coarse_aux["aux"] + refine_aux["aux"]
                    loss = ctc + 0.75 * refine_loss + 0.15 * struct_loss + 0.02 * moe_aux
                val_loss += float(loss.item())

                coarse_preds = decode_ids_to_texts(tokenizer, outputs.coarse_pred_ids, "ctc")
                final_preds = decode_ids_to_texts(tokenizer, final_refine_logits.argmax(dim=-1), "sequence")
                for coarse_pred, final_pred, ref in zip(coarse_preds, final_preds, batch["texts"]):
                    coarse_cers.append(cer(coarse_pred, ref))
                    refine_cers.append(cer(final_pred, ref))

                if preview_payload is None:
                    preview_payload = (
                        batch["preview_images"][: args.preview_count],
                        batch["texts"][: args.preview_count],
                        final_preds[: args.preview_count],
                    )

        val_loss /= max(1, len(val_loader))
        mean_coarse_cer = sum(coarse_cers) / max(1, len(coarse_cers))
        mean_refine_cer = sum(refine_cers) / max(1, len(refine_cers))
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"coarse_CER={mean_coarse_cer:.4f} refine_CER={mean_refine_cer:.4f}"
        )

        if preview_payload is not None:
            preview_path = preview_dir / f"epoch_{epoch:03d}.png"
            preview_images, refs, preds = preview_payload
            save_eval_preview(preview_images, refs, preds, preview_path, args.image_width, args.preview_count)
            print(f"saved validation preview to {preview_path}")
            for idx, (ref, pred) in enumerate(zip(refs, preds), start=1):
                print(f"sample[{idx}] ref={ref!r}")
                print(f"sample[{idx}] pred={pred!r}")

        ckpt = {
            "model": model.state_dict(),
            "config": model_cfg.__dict__,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_coarse_cer": mean_coarse_cer,
            "val_refine_cer": mean_refine_cer,
            "dataset_summary": dataset_summary,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "best.pt")

    # Export TorchScript for macOS-friendly inference
    model.eval()
    dummy = torch.randn(1, 1, args.image_height, args.image_width, device=device)
    if device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)

    class CoarseExport(nn.Module):
        def __init__(self, source_model: StrideMoEOCR):
            super().__init__()
            self.source_model = source_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.source_model.coarse_only(x)

    traced = torch.jit.trace(CoarseExport(model).eval(), dummy, check_trace=False)
    traced.save(str(output_dir / "model_ts.pt"))

    with open(output_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "batch_size": batch_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "epochs": args.epochs,
                "best_val": best_val,
                "dataset_cache_dir": str(dataset_cache_dir),
                "dataset_summary": dataset_summary,
                "preview_dir": str(preview_dir),
                "output_dir": str(output_dir),
                "runtime_info": runtime_info,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
