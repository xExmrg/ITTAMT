#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Allow running as `python scripts/train_colab.py` without installing package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
        ll = int(label_lengths[i].item())
        flat_targets.append(labels[i, :ll])
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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description="Train STRIDE-MoE OCR on Colab")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-height", type=int, default=64)
    ap.add_argument("--image-width", type=int, default=512)
    ap.add_argument("--synthetic-samples", type=int, default=40000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--output-dir", type=str, default="artifacts/stride_moe")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = CharTokenizer.build_default()
    tokenizer.save(str(Path(args.output_dir) / "tokenizer.json"))

    data_cfg = DataConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        synthetic_samples=args.synthetic_samples,
        use_iam=True,
    )
    train_loader, val_loader = build_dataloaders(tokenizer, data_cfg)

    model_cfg = StrideMoEConfig(vocab_size=len(tokenizer.itos), dim=256, depth=6, heads=8, num_experts=8, top_k=2)
    model = StrideMoEOCR(model_cfg)

    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            label_lengths = batch["label_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                label_lengths = batch["label_lengths"].to(device, non_blocking=True)

                logits, aux_loss = model(images)
                ctc = ctc_loss_from_logits(logits, labels, label_lengths, tokenizer.blank_id)
                loss = ctc + 0.05 * aux_loss
                val_loss += float(loss.item())

                preds = greedy_decode(logits, tokenizer)
                for p, r in zip(preds, batch["texts"]):
                    cers.append(cer(p, r))

        val_loss /= max(1, len(val_loader))
        mean_cer = sum(cers) / max(1, len(cers))
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CER={mean_cer:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "config": model_cfg.__dict__,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_cer": mean_cer,
        }
        torch.save(ckpt, Path(args.output_dir) / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, Path(args.output_dir) / "best.pt")

    # Export TorchScript for macOS-friendly inference
    model.eval()
    dummy = torch.randn(1, 1, args.image_height, args.image_width, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(Path(args.output_dir) / "model_ts.pt"))

    with open(Path(args.output_dir) / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump({"device": str(device), "epochs": args.epochs, "best_val": best_val}, f, indent=2)


if __name__ == "__main__":
    main()
