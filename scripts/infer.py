#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow script execution without editable install (e.g., raw Colab clone)."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

import numpy as np
import torch
from PIL import Image, ImageGrab

from ittamt.model import StrideMoEConfig, StrideMoEOCR
from ittamt.tokenizer import CharTokenizer


def load_image(path: str, width: int, height: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((width, height))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def screenshot_to_file(path: str):
    shot = ImageGrab.grab()
    shot.save(path)


def greedy_decode(ids: list[int], tokenizer: CharTokenizer) -> str:
    return tokenizer.decode_ctc(ids)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/stride_moe/best.pt")
    ap.add_argument("--tokenizer", default="artifacts/stride_moe/tokenizer.json")
    ap.add_argument("--image", default=None)
    ap.add_argument("--screenshot", action="store_true")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=64)
    args = ap.parse_args()

    if args.screenshot:
        if args.image is None:
            args.image = "tmp_screenshot.png"
        screenshot_to_file(args.image)
        print(f"Saved screenshot to {args.image}")

    if args.image is None:
        raise ValueError("Provide --image or use --screenshot")

    tokenizer = CharTokenizer.load(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = StrideMoEConfig(**ckpt["config"])
    model = StrideMoEOCR(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    device = get_device()
    model.to(device)

    x = load_image(args.image, args.width, args.height).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        ids = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()
        text = greedy_decode(ids, tokenizer)
    print(text)


if __name__ == "__main__":
    main()
