#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
import sys
import time
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow script execution without editable install (e.g., raw Colab clone)."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

import torch
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


def greedy_decode(logits: torch.Tensor, tokenizer: CharTokenizer) -> list[str]:
    ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [tokenizer.decode_ctc(row) for row in ids]


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


def default_persist_root() -> Path:
    if running_in_colab():
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            return drive_root / "ittamt"
        return Path("/content/ittamt_persist")
    return Path.home() / ".cache" / "ittamt"


def default_runtime_root() -> Path:
    if running_in_colab():
        return Path("/content/ittamt_runtime")
    return Path.cwd() / ".ittamt_runtime"


def default_dataset_cache_dir() -> str:
    return str(default_runtime_root() / "datasets")


def default_output_dir() -> str:
    return str(default_runtime_root() / "artifacts" / "stride_moe")


def default_mirror_output_dir() -> str | None:
    persist_root = default_persist_root()
    if running_in_colab() and persist_root == Path("/content/ittamt_persist"):
        return None
    return str(persist_root / "artifacts" / "stride_moe")


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


def _stage_log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def mirror_output_artifacts(output_dir: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return

    mirror_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "last.pt", "best.pt", "model_ts.pt", "train_meta.json"]:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, mirror_dir / name)

    preview_src = output_dir / "eval_previews"
    if preview_src.exists():
        shutil.copytree(preview_src, mirror_dir / "eval_previews", dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Train STRIDE-MoE OCR on Colab")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto-scale from detected GPU VRAM")
    ap.add_argument("--lr", type=float, default=3e-4)
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
    ap.add_argument("--mirror-output-dir", type=str, default=None)
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
    mirror_output_dir_arg = args.mirror_output_dir or default_mirror_output_dir()
    mirror_output_dir = Path(mirror_output_dir_arg).resolve() if mirror_output_dir_arg else None
    if mirror_output_dir is not None:
        mirror_output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  mirror_output_dir: {mirror_output_dir if mirror_output_dir is not None else 'disabled'}")
    if running_in_colab():
        print(f"  persist_root: {default_persist_root()}")

    tokenizer = CharTokenizer.build_default()
    tokenizer.save(str(output_dir / "tokenizer.json"))
    mirror_output_artifacts(output_dir, mirror_output_dir)

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
    _stage_log("building dataloaders")
    dataloader_started_at = time.perf_counter()
    train_loader, val_loader, dataset_summary = build_dataloaders(tokenizer, data_cfg)
    dataloader_elapsed = time.perf_counter() - dataloader_started_at
    _stage_log(f"dataloaders built in {dataloader_elapsed:.1f}s")
    _print_dataset_summary(dataset_summary)

    model_cfg = StrideMoEConfig(vocab_size=len(tokenizer.itos), dim=256, depth=6, heads=8, num_experts=8, top_k=2)
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

    _stage_log("preflighting first training batch")
    preflight_started_at = time.perf_counter()
    sample_batch = next(iter(train_loader))
    preflight_elapsed = time.perf_counter() - preflight_started_at
    _stage_log(
        f"first training batch ready in {preflight_elapsed:.1f}s "
        f"(images={tuple(sample_batch['image'].shape)}, labels={tuple(sample_batch['labels'].shape)})"
    )
    del sample_batch

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        _stage_log(f"starting epoch {epoch}/{args.epochs}")
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = batch["labels"].to(device, non_blocking=True)
            label_lengths = batch["label_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**autocast_kwargs):
                logits, aux_loss = model(images)
                ctc = ctc_loss_from_logits(logits, labels, label_lengths, tokenizer.blank_id)
                loss = ctc + 0.05 * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        cers = []
        preview_payload: tuple[list[Image.Image], list[str], list[str]] | None = None
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                images = batch["image"].to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = batch["labels"].to(device, non_blocking=True)
                label_lengths = batch["label_lengths"].to(device, non_blocking=True)

                with torch.autocast(**autocast_kwargs):
                    logits, aux_loss = model(images)
                    ctc = ctc_loss_from_logits(logits, labels, label_lengths, tokenizer.blank_id)
                    loss = ctc + 0.05 * aux_loss
                val_loss += float(loss.item())

                preds = greedy_decode(logits, tokenizer)
                for pred, ref in zip(preds, batch["texts"]):
                    cers.append(cer(pred, ref))

                if preview_payload is None:
                    preview_payload = (
                        batch["preview_images"][: args.preview_count],
                        batch["texts"][: args.preview_count],
                        preds[: args.preview_count],
                    )

        val_loss /= max(1, len(val_loader))
        mean_cer = sum(cers) / max(1, len(cers))
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CER={mean_cer:.4f}")

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
            "val_cer": mean_cer,
            "dataset_summary": dataset_summary,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "best.pt")

        mirror_output_artifacts(output_dir, mirror_output_dir)

    # Export TorchScript for macOS-friendly inference
    model.eval()
    dummy = torch.randn(1, 1, args.image_height, args.image_width, device=device)
    if device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    traced = torch.jit.trace(model, dummy)
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
    mirror_output_artifacts(output_dir, mirror_output_dir)


if __name__ == "__main__":
    main()
