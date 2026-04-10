from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from .tokenizer import CharTokenizer


Sample = tuple[Image.Image, str]


@dataclass
class DataConfig:
    image_height: int = 64
    image_width: int = 512
    max_label_len: int = 128
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    dataset_cache_dir: str | None = None
    synthetic_samples: int = 60000
    synthetic_val_samples: int = 3000
    use_synthetic: bool = True
    use_iam: bool = True
    use_iiit5k: bool = True
    use_sroie: bool = True
    use_cord: bool = True
    use_funsd: bool = True
    iam_train_cap: int = 18000
    iam_val_cap: int = 3000
    iiit5k_train_cap: int = 12000
    iiit5k_val_cap: int = 2000
    sroie_train_cap: int = 8000
    sroie_val_cap: int = 1500
    cord_train_cap: int = 8000
    cord_val_cap: int = 1500
    funsd_train_cap: int = 5000
    funsd_val_cap: int = 1000


class OCRDataset(Dataset):
    def __init__(self, samples: list[Sample], tokenizer: CharTokenizer, cfg: DataConfig):
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
            "preview_image": img.convert("RGB"),
        }


def _collate(batch: list[dict[str, Any]], pad_id: int):
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    max_len = max([len(l) for l in labels] + [1])
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, label in enumerate(labels):
        if len(label) > 0:
            padded[i, : len(label)] = label
    return {
        "image": images,
        "labels": padded,
        "label_lengths": lengths,
        "texts": [b["text"] for b in batch],
        "preview_images": [b["preview_image"] for b in batch],
    }


def _normalize_text(text: str, preserve_newlines: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(part.strip().split()) for part in text.split("\n")]
    lines = [line for line in lines if line]
    if preserve_newlines:
        return "\n".join(lines)
    return " ".join(lines)


def _safe_get_text(example: dict[str, Any]) -> str:
    for key in ["text", "sentence", "label", "transcription", "ground_truth"]:
        if key in example and isinstance(example[key], str):
            return example[key]
    return ""


def _safe_get_image(example: dict[str, Any]) -> Image.Image | None:
    if "image" in example:
        image = example["image"]
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        try:
            return image.convert("RGB")
        except Exception:
            return None
    for key in ["img", "pixel_values"]:
        if key in example and isinstance(example[key], Image.Image):
            return example[key].convert("RGB")
    return None


def _record_count(summary: dict[str, Any], split_name: str, dataset_name: str, count: int) -> None:
    summary.setdefault(split_name, {})
    summary[split_name][dataset_name] = count


def _record_warning(summary: dict[str, Any], dataset_name: str, exc: Exception) -> None:
    summary.setdefault("warnings", [])
    summary["warnings"].append(f"{dataset_name}: {type(exc).__name__}: {exc}")


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    xs0 = [box[0] for box in boxes]
    ys0 = [box[1] for box in boxes]
    xs1 = [box[2] for box in boxes]
    ys1 = [box[3] for box in boxes]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _resolve_bbox(box: list[int] | tuple[int, int, int, int], image: Image.Image) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 > image.width + 5 or y1 > image.height + 5:
        sx = image.width / 1000.0
        sy = image.height / 1000.0
        x0 = int(round(x0 * sx))
        x1 = int(round(x1 * sx))
        y0 = int(round(y0 * sy))
        y1 = int(round(y1 * sy))
    return x0, y0, x1, y1


def _quad_to_bbox(words: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for word in words:
        quad = word.get("quad", {})
        xs.extend(int(quad.get(key, 0)) for key in ["x1", "x2", "x3", "x4"])
        ys.extend(int(quad.get(key, 0)) for key in ["y1", "y2", "y3", "y4"])
    return min(xs), min(ys), max(xs), max(ys)


def _crop_image(image: Image.Image, box: tuple[int, int, int, int], pad: int = 6) -> Image.Image | None:
    width, height = image.size
    x0 = max(0, box[0] - pad)
    y0 = max(0, box[1] - pad)
    x1 = min(width, box[2] + pad)
    y1 = min(height, box[3] + pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def _token_height(box: tuple[int, int, int, int]) -> int:
    return max(1, box[3] - box[1])


def _group_tokens_into_lines(tokens: list[tuple[str, tuple[int, int, int, int]]]) -> list[tuple[str, tuple[int, int, int, int]]]:
    if not tokens:
        return []

    heights = sorted(_token_height(box) for _, box in tokens)
    median_height = heights[len(heights) // 2]
    line_tolerance = max(8, int(median_height * 0.65))

    rows: list[dict[str, Any]] = []
    for text, box in sorted(tokens, key=lambda item: ((item[1][1] + item[1][3]) / 2.0, item[1][0])):
        center_y = (box[1] + box[3]) / 2.0
        if rows and abs(center_y - rows[-1]["center_y"]) <= line_tolerance:
            rows[-1]["tokens"].append((text, box))
            rows[-1]["center_y"] = (rows[-1]["center_y"] + center_y) / 2.0
        else:
            rows.append({"center_y": center_y, "tokens": [(text, box)]})

    line_samples: list[tuple[str, tuple[int, int, int, int]]] = []
    for row in rows:
        row_tokens = sorted(row["tokens"], key=lambda item: item[1][0])
        row_text = _normalize_text(" ".join(text for text, _ in row_tokens))
        if len(row_text) < 2:
            continue
        row_box = _union_bbox([box for _, box in row_tokens])
        line_samples.append((row_text, row_box))
    return line_samples


def _extract_line_crops(
    image: Image.Image,
    tokens: list[tuple[str, tuple[int, int, int, int]]],
    limit: int,
) -> list[Sample]:
    samples: list[Sample] = []
    for text, box in _group_tokens_into_lines(tokens):
        crop = _crop_image(image, box)
        if crop is None:
            continue
        samples.append((crop, text))
        if len(samples) >= limit:
            break
    return samples


def _structured_synthetic_text() -> str:
    invoice_id = random.randint(1000, 9999)
    total_value = f"{random.randint(8, 299)}.{random.randint(0, 99):02d}"
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    phone = f"+49 30 {random.randint(100000, 999999)}"
    templates = [
        f"Invoice #{invoice_id}\nDate: 2026-{month:02d}-{day:02d}\nTotal: EUR {total_value}",
        f"Ship To:\nMarta Stein\nBergstrasse {random.randint(1, 99)}\n10115 Berlin",
        f"2 x Latte   9.50\n1 x Bagel   4.20\nTOTAL {total_value}",
        f"Meeting Notes\nRoom B-12\nCall me at {phone}",
        f"Order {invoice_id} | Qty 3 | Paid {total_value}",
        f"Name: Alex Rivera\nRef: AB-{invoice_id}\nStatus: APPROVED",
    ]
    return random.choice(templates)


def make_synthetic_samples(n: int, width: int, height: int) -> list[Sample]:
    plain_phrases = [
        "The quick brown fox jumps over 13 lazy dogs.",
        "Invoice A-2048 paid on 2026-01-17.",
        "Handwritten notes can be messy but meaningful.",
        "AI OCR should be fast, accurate, and robust.",
        "Call me at 555-0199 after 7:30 PM!",
        "Structure-aware decoding improves readability.",
    ]
    font = ImageFont.load_default()
    samples: list[Sample] = []
    for _ in range(n):
        text = _structured_synthetic_text() if random.random() < 0.65 else random.choice(plain_phrases)
        preserve_newlines = "\n" in text
        canvas_height = height if not preserve_newlines else height * random.randint(2, 3)
        image = Image.new("L", (width, canvas_height), color=255)
        draw = ImageDraw.Draw(image)

        if random.random() < 0.5:
            for y in range(0, canvas_height, max(12, canvas_height // 6)):
                draw.line((0, y, width, y), fill=245, width=1)

        x = random.randint(6, 20)
        y = random.randint(6, max(7, canvas_height // 6))
        fill = random.randint(0, 50)
        if preserve_newlines:
            draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4)
        else:
            draw.text((x, y), text, font=font, fill=fill)
        samples.append((image.convert("RGB"), _normalize_text(text, preserve_newlines=preserve_newlines)))
    return samples


def _load_first_available_direct(
    dataset_names: list[str],
    split: str,
    cap: int,
    summary: dict[str, Any],
    summary_name: str,
    cache_dir: str | None = None,
) -> list[Sample]:
    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    samples: list[Sample] = []
    for dataset_name in dataset_names:
        try:
            ds = load_dataset(dataset_name, split=split_map[split], cache_dir=cache_dir)
            for example in ds:
                text = _normalize_text(_safe_get_text(example), preserve_newlines=True)
                image = _safe_get_image(example)
                if text and image is not None:
                    samples.append((image, text))
                if len(samples) >= cap:
                    break
            if samples:
                _record_count(summary, split, summary_name, len(samples))
                return samples
        except Exception as exc:
            _record_warning(summary, dataset_name, exc)
    _record_count(summary, split, summary_name, 0)
    return samples


def load_iiit5k_split(split: str, cap: int, summary: dict[str, Any], cache_dir: str | None = None) -> list[Sample]:
    resolved_split = "test" if split == "validation" else split
    return _load_first_available_direct(
        ["MiXaiLL76/IIIT5K_OCR"],
        resolved_split,
        cap,
        summary,
        "iiit5k",
        cache_dir=cache_dir,
    )


def load_iam_split(split: str, cap: int, summary: dict[str, Any], cache_dir: str | None = None) -> list[Sample]:
    dataset_names = ["Teklia/IAM-line", "Teklia/IAM-lines", "LarsHill/IAM-lines"]
    resolved_split = "validation" if split == "validation" else split
    return _load_first_available_direct(
        dataset_names,
        resolved_split,
        cap,
        summary,
        "iam",
        cache_dir=cache_dir,
    )


def load_sroie_split(split: str, cap: int, summary: dict[str, Any], cache_dir: str | None = None) -> list[Sample]:
    dataset_name = "jsdnrs/ICDAR2019-SROIE"
    resolved_split = "test" if split == "validation" else split
    try:
        ds = load_dataset(dataset_name, split=resolved_split, cache_dir=cache_dir)
        samples: list[Sample] = []
        for example in ds:
            image = _safe_get_image(example)
            if image is None:
                continue
            tokens: list[tuple[str, tuple[int, int, int, int]]] = []
            for word, box in zip(example.get("words", []), example.get("bboxes", [])):
                text = _normalize_text(str(word))
                if not text:
                    continue
                tokens.append((text, _resolve_bbox(box, image)))
            samples.extend(_extract_line_crops(image, tokens, max(0, cap - len(samples))))
            if len(samples) >= cap:
                break
        _record_count(summary, split, "sroie", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, dataset_name, exc)
        _record_count(summary, split, "sroie", 0)
        return []


def load_cord_split(split: str, cap: int, summary: dict[str, Any], cache_dir: str | None = None) -> list[Sample]:
    dataset_name = "naver-clova-ix/cord-v2"
    try:
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        samples: list[Sample] = []
        for example in ds:
            image = _safe_get_image(example)
            gt_raw = example.get("ground_truth")
            if image is None or not isinstance(gt_raw, str):
                continue
            gt = json.loads(gt_raw)
            groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for line in gt.get("valid_line", []):
                group_key = (int(line.get("group_id", -1)), int(line.get("sub_group_id", 0)))
                groups.setdefault(group_key, []).append(line)

            for grouped_lines in groups.values():
                words: list[dict[str, Any]] = []
                for line in grouped_lines:
                    words.extend(line.get("words", []))
                if not words:
                    continue
                words = sorted(words, key=lambda item: (min(item["quad"].get(key, 0) for key in ["y1", "y2", "y3", "y4"]), min(item["quad"].get(key, 0) for key in ["x1", "x2", "x3", "x4"])))
                text = _normalize_text(" ".join(word.get("text", "") for word in words))
                if len(text) < 2:
                    continue
                crop = _crop_image(image, _quad_to_bbox(words))
                if crop is None:
                    continue
                samples.append((crop, text))
                if len(samples) >= cap:
                    break
            if len(samples) >= cap:
                break
        _record_count(summary, split, "cord", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, dataset_name, exc)
        _record_count(summary, split, "cord", 0)
        return []


def load_funsd_split(split: str, cap: int, summary: dict[str, Any], cache_dir: str | None = None) -> list[Sample]:
    dataset_name = "nielsr/funsd"
    resolved_split = "test" if split == "validation" else split
    try:
        ds = load_dataset(dataset_name, split=resolved_split, cache_dir=cache_dir)
        samples: list[Sample] = []
        for example in ds:
            image = _safe_get_image(example)
            if image is None:
                continue
            tokens: list[tuple[str, tuple[int, int, int, int]]] = []
            for word, box in zip(example.get("words", []), example.get("bboxes", [])):
                text = _normalize_text(str(word))
                if not text:
                    continue
                tokens.append((text, _resolve_bbox(box, image)))
            samples.extend(_extract_line_crops(image, tokens, max(0, cap - len(samples))))
            if len(samples) >= cap:
                break
        _record_count(summary, split, "funsd", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, dataset_name, exc)
        _record_count(summary, split, "funsd", 0)
        return []


def build_dataloaders(tokenizer: CharTokenizer, cfg: DataConfig):
    summary: dict[str, Any] = {"train": {}, "validation": {}, "warnings": []}
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    if cfg.use_synthetic:
        synthetic_train = make_synthetic_samples(cfg.synthetic_samples, cfg.image_width, cfg.image_height)
        synthetic_val = make_synthetic_samples(cfg.synthetic_val_samples, cfg.image_width, cfg.image_height)
        train_samples.extend(synthetic_train)
        val_samples.extend(synthetic_val)
        _record_count(summary, "train", "synthetic", len(synthetic_train))
        _record_count(summary, "validation", "synthetic", len(synthetic_val))

    if cfg.use_iam:
        train_samples.extend(load_iam_split("train", cfg.iam_train_cap, summary, cache_dir=cfg.dataset_cache_dir))
        val_samples.extend(load_iam_split("validation", cfg.iam_val_cap, summary, cache_dir=cfg.dataset_cache_dir))

    if cfg.use_iiit5k:
        train_samples.extend(load_iiit5k_split("train", cfg.iiit5k_train_cap, summary, cache_dir=cfg.dataset_cache_dir))
        val_samples.extend(load_iiit5k_split("validation", cfg.iiit5k_val_cap, summary, cache_dir=cfg.dataset_cache_dir))

    if cfg.use_sroie:
        train_samples.extend(load_sroie_split("train", cfg.sroie_train_cap, summary, cache_dir=cfg.dataset_cache_dir))
        val_samples.extend(load_sroie_split("validation", cfg.sroie_val_cap, summary, cache_dir=cfg.dataset_cache_dir))

    if cfg.use_cord:
        train_samples.extend(load_cord_split("train", cfg.cord_train_cap, summary, cache_dir=cfg.dataset_cache_dir))
        val_samples.extend(load_cord_split("validation", cfg.cord_val_cap, summary, cache_dir=cfg.dataset_cache_dir))

    if cfg.use_funsd:
        train_samples.extend(load_funsd_split("train", cfg.funsd_train_cap, summary, cache_dir=cfg.dataset_cache_dir))
        val_samples.extend(load_funsd_split("validation", cfg.funsd_val_cap, summary, cache_dir=cfg.dataset_cache_dir))

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_ds = OCRDataset(train_samples, tokenizer, cfg)
    val_ds = OCRDataset(val_samples, tokenizer, cfg)

    collate_fn = lambda batch: _collate(batch, pad_id=tokenizer.blank_id)
    loader_kwargs: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "collate_fn": collate_fn,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(2, cfg.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, summary
