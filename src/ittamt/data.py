from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from .tokenizer import CharTokenizer


@dataclass
class DataConfig:
    image_height: int = 64
    image_width: int = 512
    max_label_len: int = 128
    batch_size: int = 16
    num_workers: int = 2
    synthetic_samples: int = 60000
    use_iam: bool = True


class OCRDataset(Dataset):
    def __init__(self, samples: list[tuple[Image.Image, str]], tokenizer: CharTokenizer, cfg: DataConfig):
        self.samples = samples
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def _prep_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L").resize((self.cfg.image_width, self.cfg.image_height))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img, text = self.samples[idx]
        label = self.tokenizer.encode(text)[: self.cfg.max_label_len]
        return {
            "image": self._prep_image(img),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }


def _collate(batch: list[dict[str, Any]], pad_id: int):
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    max_len = max([len(l) for l in labels] + [1])
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, l in enumerate(labels):
        if len(l) > 0:
            padded[i, : len(l)] = l
    return {
        "image": images,
        "labels": padded,
        "label_lengths": lengths,
        "texts": [b["text"] for b in batch],
    }


def _safe_get_text(example: dict[str, Any]) -> str:
    for key in ["text", "sentence", "label", "transcription", "ground_truth"]:
        if key in example and isinstance(example[key], str):
            return example[key]
    return ""


def _safe_get_image(example: dict[str, Any]) -> Image.Image | None:
    if "image" in example:
        img = example["image"]
        if isinstance(img, Image.Image):
            return img
        try:
            return img.convert("RGB")
        except Exception:
            return None
    for key in ["img", "pixel_values"]:
        if key in example and isinstance(example[key], Image.Image):
            return example[key]
    return None


def make_synthetic_samples(n: int, width: int, height: int) -> list[tuple[Image.Image, str]]:
    phrases = [
        "The quick brown fox jumps over 13 lazy dogs.",
        "Invoice #A-2048 paid on 2026-01-17.",
        "Handwritten notes can be messy but meaningful.",
        "AI OCR should be fast, accurate, and robust.",
        "Call me at 555-0199 after 7:30 PM!",
        "Structure-aware decoding improves readability.",
    ]
    font = ImageFont.load_default()
    out = []
    for _ in range(n):
        text = random.choice(phrases)
        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)
        x = random.randint(6, 24)
        y = random.randint(6, max(7, height // 3))
        draw.text((x, y), text, font=font, fill=random.randint(0, 40))
        out.append((image.convert("RGB"), text))
    return out


def load_iam_like_split(split: str, cap: int = 12000) -> list[tuple[Image.Image, str]]:
    """
    Tries a public IAM-like dataset on HF Hub.
    Falls back cleanly if unavailable.
    """
    candidates = [
        "Teklia/IAM-lines",
        "LarsHill/IAM-lines",
    ]
    for ds_name in candidates:
        try:
            ds = load_dataset(ds_name, split=split)
            samples = []
            for ex in ds:
                text = _safe_get_text(ex)
                img = _safe_get_image(ex)
                if text and img is not None:
                    samples.append((img, text))
                if len(samples) >= cap:
                    break
            if samples:
                return samples
        except Exception:
            continue
    return []


def build_dataloaders(tokenizer: CharTokenizer, cfg: DataConfig):
    train_samples = make_synthetic_samples(cfg.synthetic_samples, cfg.image_width, cfg.image_height)
    val_samples = make_synthetic_samples(max(2000, cfg.synthetic_samples // 20), cfg.image_width, cfg.image_height)

    if cfg.use_iam:
        train_iam = load_iam_like_split("train", cap=18000)
        test_iam = load_iam_like_split("test", cap=3000)
        if train_iam:
            train_samples.extend(train_iam)
        if test_iam:
            val_samples.extend(test_iam)

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_ds = OCRDataset(train_samples, tokenizer, cfg)
    val_ds = OCRDataset(val_samples, tokenizer, cfg)

    collate_fn = lambda b: _collate(b, pad_id=tokenizer.blank_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
