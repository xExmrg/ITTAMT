from __future__ import annotations

import json
import random
import signal
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .tokenizer import CharTokenizer


Sample = tuple[Image.Image, str]

_XFUND_BASE_URL = "https://github.com/doc-analysis/XFUN/releases/download/v1.0/"
_XFUND_LANGS = ("zh", "de", "es", "fr", "it", "ja", "pt")


@dataclass
class DataConfig:
    image_height: int = 64
    image_width: int = 512
    max_label_len: int = 128
    batch_size: int = 16
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    dataset_cache_dir: str | None = None
    synthetic_samples: int = 60000
    synthetic_val_samples: int = 3000
    use_synthetic: bool = True
    use_iam: bool = True
    use_iiit5k: bool = True
    use_textocr: bool = True
    use_sroie: bool = True
    use_cord: bool = True
    use_funsd: bool = True
    use_doclaynet: bool = True
    use_xfund: bool = True
    iam_train_cap: int = 18000
    iam_val_cap: int = 3000
    iiit5k_train_cap: int = 12000
    iiit5k_val_cap: int = 2000
    textocr_train_cap: int = 24000
    textocr_val_cap: int = 3500
    sroie_train_cap: int = 8000
    sroie_val_cap: int = 1500
    cord_train_cap: int = 8000
    cord_val_cap: int = 1500
    funsd_train_cap: int = 5000
    funsd_val_cap: int = 1000
    doclaynet_train_cap: int = 18000
    doclaynet_val_cap: int = 3000
    xfund_train_cap: int = 14000
    xfund_val_cap: int = 2500
    xfund_languages: tuple[str, ...] = _XFUND_LANGS


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

    def _prep_map01(self, image: Image.Image) -> torch.Tensor:
        """Prepare a [1, H, W] float map in [0, 1] aligned with model inputs."""
        image = image.convert("L").resize((self.cfg.image_width, self.cfg.image_height))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if isinstance(sample, tuple) and len(sample) == 3:
            img, text, struct = sample
        else:
            img, text = sample  # type: ignore[misc]
            struct = None

        # CTC training target (no BOS/EOS).
        label = self.tokenizer.encode(text)[: self.cfg.max_label_len]
        # Optional refinement target (BOS/EOS + padding).
        seq_label = self.tokenizer.encode_sequence(text, max_length=self.cfg.max_label_len + 2)

        out: dict[str, Any] = {
            "image": self._prep_image(img),
            "label": torch.tensor(label, dtype=torch.long),
            "seq_label": torch.tensor(seq_label, dtype=torch.long),
            "text": text,
            "preview_image": img.convert("RGB"),
        }

        if isinstance(struct, dict):
            text_mask = struct.get("text_mask")
            baseline = struct.get("baseline_heatmap")
            centers = struct.get("char_center_heatmap")
            if isinstance(text_mask, Image.Image) and isinstance(baseline, Image.Image) and isinstance(centers, Image.Image):
                text_mask_t = self._prep_map01(text_mask)
                baseline_t = self._prep_map01(baseline)
                centers_t = self._prep_map01(centers)
                density = text_mask_t.squeeze(0).sum(dim=0) / max(1.0, float(text_mask_t.shape[1]))  # [W]
                out["struct"] = {
                    "text_mask": text_mask_t,
                    "baseline_heatmap": baseline_t,
                    "char_center_heatmap": centers_t,
                    "density": density.to(dtype=torch.float32),
                }
                out["struct_valid"] = True
            else:
                out["struct"] = None
                out["struct_valid"] = False
        else:
            out["struct"] = None
            out["struct_valid"] = False

        return out


class SyntheticOCRDataset(Dataset):
    def __init__(self, size: int, tokenizer: CharTokenizer, cfg: DataConfig, seed: int):
        self.size = size
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = _make_synthetic_sample(
            width=self.cfg.image_width,
            height=self.cfg.image_height,
            rng=random.Random(self.seed + idx),
        )
        return OCRDataset([sample], self.tokenizer, self.cfg)[0]


def _data_log(message: str) -> None:
    print(f"[data] {message}", flush=True)


def _collate(batch: list[dict[str, Any]], pad_id: int, seq_pad_id: int | None = None):
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    max_len = max([len(l) for l in labels] + [1])
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, label in enumerate(labels):
        if len(label) > 0:
            padded[i, : len(label)] = label

    seq_labels = [b.get("seq_label", torch.tensor([], dtype=torch.long)) for b in batch]
    seq_lengths = torch.tensor([len(l) for l in seq_labels], dtype=torch.long)
    seq_max_len = max([len(l) for l in seq_labels] + [1])
    effective_seq_pad = pad_id if seq_pad_id is None else seq_pad_id
    seq_padded = torch.full((len(batch), seq_max_len), effective_seq_pad, dtype=torch.long)
    for i, label in enumerate(seq_labels):
        if len(label) > 0:
            seq_padded[i, : len(label)] = label

    # Structure supervision (available for synthetic samples). Real datasets set struct_valid=0.
    h = int(images.shape[2])
    w = int(images.shape[3])
    struct_valid = torch.tensor([bool(b.get("struct_valid", False)) for b in batch], dtype=torch.bool)
    text_mask = []
    baseline = []
    centers = []
    density = []
    for b in batch:
        struct = b.get("struct")
        if isinstance(struct, dict):
            text_mask.append(struct.get("text_mask", torch.zeros((1, h, w), dtype=torch.float32)))
            baseline.append(struct.get("baseline_heatmap", torch.zeros((1, h, w), dtype=torch.float32)))
            centers.append(struct.get("char_center_heatmap", torch.zeros((1, h, w), dtype=torch.float32)))
            density.append(struct.get("density", torch.zeros((w,), dtype=torch.float32)))
        else:
            text_mask.append(torch.zeros((1, h, w), dtype=torch.float32))
            baseline.append(torch.zeros((1, h, w), dtype=torch.float32))
            centers.append(torch.zeros((1, h, w), dtype=torch.float32))
            density.append(torch.zeros((w,), dtype=torch.float32))
    return {
        "image": images,
        "labels": padded,
        "label_lengths": lengths,
        "seq_labels": seq_padded,
        "seq_label_lengths": seq_lengths,
        "texts": [b["text"] for b in batch],
        "preview_images": [b["preview_image"] for b in batch],
        "struct": {
            "text_mask": torch.stack(text_mask, dim=0),
            "baseline_heatmap": torch.stack(baseline, dim=0),
            "char_center_heatmap": torch.stack(centers, dim=0),
            "density": torch.stack(density, dim=0),
            "valid": struct_valid,
        },
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
    for key in ["img", "pixel_values", "original_image"]:
        if key in example:
            image = example[key]
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            try:
                return image.convert("RGB")
            except Exception:
                continue
    return None


def _record_count(summary: dict[str, Any], split_name: str, dataset_name: str, count: int) -> None:
    summary.setdefault(split_name, {})
    summary[split_name][dataset_name] = count


def _record_warning(summary: dict[str, Any], dataset_name: str, exc: Exception | str) -> None:
    summary.setdefault("warnings", [])
    if isinstance(exc, Exception):
        summary["warnings"].append(f"{dataset_name}: {type(exc).__name__}: {exc}")
    else:
        summary["warnings"].append(f"{dataset_name}: {exc}")


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


def _resolve_xywh_bbox(box: list[float] | tuple[float, float, float, float], image: Image.Image) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in box]
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    if max(x + w, y + h) > max(image.width, image.height) + 5:
        sx = image.width / 1025.0
        sy = image.height / 1025.0
        x *= sx
        y *= sy
        w *= sx
        h *= sy
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def _quad_to_bbox(words: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for word in words:
        quad = word.get("quad", {})
        xs.extend(int(quad.get(key, 0)) for key in ["x1", "x2", "x3", "x4"])
        ys.extend(int(quad.get(key, 0)) for key in ["y1", "y2", "y3", "y4"])
    return min(xs), min(ys), max(xs), max(ys)


def _simplify_quad_bbox(box: list[int] | tuple[int, ...]) -> tuple[int, int, int, int]:
    xs = [int(v) for v in box[0::2]]
    ys = [int(v) for v in box[1::2]]
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


def _structured_synthetic_text(rng: random.Random) -> str:
    invoice_id = rng.randint(1000, 9999)
    total_value = f"{rng.randint(8, 299)}.{rng.randint(0, 99):02d}"
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    phone = f"+49 30 {rng.randint(100000, 999999)}"
    templates = [
        f"Invoice #{invoice_id}\nDate: 2026-{month:02d}-{day:02d}\nTotal: EUR {total_value}",
        f"Ship To:\nMarta Stein\nBergstrasse {rng.randint(1, 99)}\n10115 Berlin",
        f"2 x Latte   9.50\n1 x Bagel   4.20\nTOTAL {total_value}",
        f"Meeting Notes\nRoom B-12\nCall me at {phone}",
        f"Order {invoice_id} | Qty 3 | Paid {total_value}",
        f"Name: Alex Rivera\nRef: AB-{invoice_id}\nStatus: APPROVED",
    ]
    return rng.choice(templates)


def _make_synthetic_sample(width: int, height: int, rng: random.Random, font: ImageFont.ImageFont | None = None) -> Sample:
    plain_phrases = [
        "The quick brown fox jumps over 13 lazy dogs.",
        "Invoice A-2048 paid on 2026-01-17.",
        "Handwritten notes can be messy but meaningful.",
        "AI OCR should be fast, accurate, and robust.",
        "Call me at 555-0199 after 7:30 PM!",
        "Structure-aware decoding improves readability.",
    ]
    if font is None:
        font = ImageFont.load_default()

    text = _structured_synthetic_text(rng) if rng.random() < 0.65 else rng.choice(plain_phrases)
    preserve_newlines = "\n" in text
    spacing = 4
    canvas_height = height if not preserve_newlines else height * rng.randint(2, 3)
    image = Image.new("L", (width, canvas_height), color=255)
    draw = ImageDraw.Draw(image)

    if rng.random() < 0.5:
        for y in range(0, canvas_height, max(12, canvas_height // 6)):
            draw.line((0, y, width, y), fill=245, width=1)

    x = rng.randint(6, 20)
    y = rng.randint(6, max(7, canvas_height // 6))
    fill = rng.randint(0, 50)
    norm_text = _normalize_text(text, preserve_newlines=preserve_newlines)

    text_mask = Image.new("L", (width, canvas_height), color=0)
    mask_draw = ImageDraw.Draw(text_mask)
    baseline_map = Image.new("L", (width, canvas_height), color=0)
    base_draw = ImageDraw.Draw(baseline_map)
    centers_map = Image.new("L", (width, canvas_height), color=0)
    centers_draw = ImageDraw.Draw(centers_map)

    try:
        ascent, descent = font.getmetrics()
    except Exception:
        ascent, descent = 10, 3
    line_h = int(ascent + descent + spacing)

    def _text_length(s: str) -> float:
        try:
            return float(draw.textlength(s, font=font))
        except Exception:
            try:
                return float(font.getsize(s)[0])
            except Exception:
                return float(len(s) * 8)

    if preserve_newlines:
        draw.multiline_text((x, y), norm_text, font=font, fill=fill, spacing=spacing)
        mask_draw.multiline_text((x, y), norm_text, font=font, fill=255, spacing=spacing)
        lines = norm_text.split("\n")
    else:
        draw.text((x, y), norm_text, font=font, fill=fill)
        mask_draw.text((x, y), norm_text, font=font, fill=255)
        lines = [norm_text]

    for li, line in enumerate(lines):
        baseline_y = float(y + ascent + li * line_h)
        line_w = _text_length(line)
        base_draw.line((x, baseline_y, x + line_w, baseline_y), fill=255, width=1)

        for ci, ch in enumerate(line):
            if ch == " ":
                continue
            prefix = line[:ci]
            ch_w = _text_length(ch)
            cx = float(x) + _text_length(prefix) + ch_w / 2.0
            cy = baseline_y - float(ascent) / 2.0
            r = 1
            centers_draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)

    baseline_map = baseline_map.filter(ImageFilter.GaussianBlur(radius=1.25))
    centers_map = centers_map.filter(ImageFilter.GaussianBlur(radius=1.5))

    return (
        image.convert("RGB"),
        norm_text,
        {
            "text_mask": text_mask,
            "baseline_heatmap": baseline_map,
            "char_center_heatmap": centers_map,
        },
    )


def make_synthetic_samples(n: int, width: int, height: int) -> list[Sample]:
    font = ImageFont.load_default()
    samples: list[Sample] = []
    for idx in range(n):
        samples.append(_make_synthetic_sample(width=width, height=height, rng=random.Random(1337 + idx), font=font))
    return samples


def _cache_dir(cfg: DataConfig) -> str | None:
    return cfg.dataset_cache_dir


class _DatasetLoadTimeout(Exception):
    pass


def _load_dataset_logged(dataset_name: str, split: str, cfg: DataConfig, timeout_sec: int | None = None):
    _data_log(f"load_dataset start dataset={dataset_name} split={split}")
    started_at = time.perf_counter()

    previous_handler = None

    def _timeout_handler(signum, frame):  # type: ignore[unused-argument]
        raise _DatasetLoadTimeout(
            f"load_dataset timed out after {timeout_sec}s for dataset={dataset_name} split={split}"
        )

    try:
        if timeout_sec is not None and hasattr(signal, "SIGALRM"):
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)
        ds = load_dataset(dataset_name, split=split, cache_dir=_cache_dir(cfg))
    finally:
        if timeout_sec is not None and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    elapsed = time.perf_counter() - started_at
    try:
        row_count = len(ds)
    except Exception:
        row_count = "unknown"
    _data_log(f"load_dataset done dataset={dataset_name} split={split} rows={row_count} in {elapsed:.1f}s")
    return ds


def _maybe_log_scan_progress(stage: str, seen: int, kept: int, started_at: float, every: int = 1000) -> None:
    if seen == 1 or seen % every == 0:
        elapsed = time.perf_counter() - started_at
        _data_log(f"{stage}: scanned={seen} kept={kept} elapsed={elapsed:.1f}s")


def _load_direct_samples(
    dataset_names: list[str],
    splits: list[str],
    cap: int,
    summary: dict[str, Any],
    summary_name: str,
    cfg: DataConfig,
) -> list[Sample]:
    samples: list[Sample] = []
    for dataset_name in dataset_names:
        try:
            for split_name in splits:
                ds = _load_dataset_logged(dataset_name, split_name, cfg)
                iter_started_at = time.perf_counter()
                for seen, example in enumerate(ds, start=1):
                    text = _normalize_text(_safe_get_text(example), preserve_newlines=True)
                    image = _safe_get_image(example)
                    if text and image is not None:
                        samples.append((image, text))
                    _maybe_log_scan_progress(f"{summary_name}/{split_name}", seen, len(samples), iter_started_at)
                    if len(samples) >= cap:
                        break
                if len(samples) >= cap:
                    break
            if samples:
                _record_count(summary, "train" if "train" in splits[0] else "validation", summary_name, len(samples))
                return samples
        except Exception as exc:
            _record_warning(summary, dataset_name, exc)
    split_name = "train" if "train" in splits[0] else "validation"
    _record_count(summary, split_name, summary_name, 0)
    return samples


def load_iam_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    dataset_names = ["Teklia/IAM-line", "Teklia/IAM-lines", "LarsHill/IAM-lines"]
    resolved_split = "validation" if split == "validation" else split
    samples: list[Sample] = []
    for dataset_name in dataset_names:
        try:
            ds = _load_dataset_logged(dataset_name, resolved_split, cfg, timeout_sec=90)
            iter_started_at = time.perf_counter()
            for seen, example in enumerate(ds, start=1):
                text = _normalize_text(_safe_get_text(example), preserve_newlines=True)
                image = _safe_get_image(example)
                if text and image is not None:
                    samples.append((image, text))
                _maybe_log_scan_progress(f"iam/{resolved_split}", seen, len(samples), iter_started_at)
                if len(samples) >= cap:
                    break
            if samples:
                _record_count(summary, split, "iam", len(samples))
                return samples
        except Exception as exc:
            _record_warning(summary, dataset_name, exc)
    _record_count(summary, split, "iam", 0)
    return []


def load_iiit5k_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    split_names = ["train", "train_numbers"] if split == "train" else ["test", "test_numbers"]
    samples: list[Sample] = []
    try:
        for split_name in split_names:
            ds = _load_dataset_logged("MiXaiLL76/IIIT5K_OCR", split_name, cfg)
            iter_started_at = time.perf_counter()
            for seen, example in enumerate(ds, start=1):
                text = _normalize_text(_safe_get_text(example))
                image = _safe_get_image(example)
                if text and image is not None:
                    samples.append((image, text))
                _maybe_log_scan_progress(f"iiit5k/{split_name}", seen, len(samples), iter_started_at)
                if len(samples) >= cap:
                    break
            if len(samples) >= cap:
                break
        _record_count(summary, split, "iiit5k", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, "MiXaiLL76/IIIT5K_OCR", exc)
        _record_count(summary, split, "iiit5k", 0)
        return []


def load_textocr_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    split_names = ["train", "train_numbers"] if split == "train" else ["test", "test_numbers"]
    samples: list[Sample] = []
    try:
        for split_name in split_names:
            ds = _load_dataset_logged("MiXaiLL76/TextOCR_OCR", split_name, cfg)
            iter_started_at = time.perf_counter()
            for seen, example in enumerate(ds, start=1):
                text = _normalize_text(_safe_get_text(example))
                image = _safe_get_image(example)
                if text and image is not None:
                    samples.append((image, text))
                _maybe_log_scan_progress(f"textocr/{split_name}", seen, len(samples), iter_started_at)
                if len(samples) >= cap:
                    break
            if len(samples) >= cap:
                break
        _record_count(summary, split, "textocr", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, "MiXaiLL76/TextOCR_OCR", exc)
        _record_count(summary, split, "textocr", 0)
        return []


def load_sroie_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    dataset_candidates = [
        ("rth/sroie-2019-v2", "objects"),
        ("jsdnrs/ICDAR2019-SROIE", "words"),
    ]
    resolved_split = "test" if split == "validation" else split
    for dataset_name, mode in dataset_candidates:
        try:
            ds = _load_dataset_logged(dataset_name, resolved_split, cfg)
            samples: list[Sample] = []
            iter_started_at = time.perf_counter()
            for seen, example in enumerate(ds, start=1):
                image = _safe_get_image(example)
                if image is None:
                    _maybe_log_scan_progress(f"sroie/{dataset_name}/{resolved_split}", seen, len(samples), iter_started_at)
                    continue
                tokens: list[tuple[str, tuple[int, int, int, int]]] = []
                if mode == "objects":
                    objects = example.get("objects", {})
                    for word, box in zip(objects.get("text", []), objects.get("bbox", [])):
                        text = _normalize_text(str(word))
                        if not text:
                            continue
                        if len(box) != 4:
                            continue
                        x0, y0, a, b = [int(v) for v in box]
                        if a > x0 and b > y0:
                            bbox = _resolve_bbox((x0, y0, a, b), image)
                        else:
                            bbox = (x0, y0, x0 + a, y0 + b)
                        tokens.append((text, bbox))
                else:
                    for word, box in zip(example.get("words", []), example.get("bboxes", [])):
                        text = _normalize_text(str(word))
                        if not text:
                            continue
                        tokens.append((text, _resolve_bbox(box, image)))
                samples.extend(_extract_line_crops(image, tokens, max(0, cap - len(samples))))
                _maybe_log_scan_progress(f"sroie/{dataset_name}/{resolved_split}", seen, len(samples), iter_started_at)
                if len(samples) >= cap:
                    break
            _record_count(summary, split, "sroie", len(samples))
            if samples:
                return samples
        except Exception as exc:
            _record_warning(summary, dataset_name, exc)
    _record_count(summary, split, "sroie", 0)
    return []


def load_cord_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    dataset_name = "naver-clova-ix/cord-v2"
    try:
        ds = _load_dataset_logged(dataset_name, split, cfg)
        samples: list[Sample] = []
        iter_started_at = time.perf_counter()
        for seen, example in enumerate(ds, start=1):
            image = _safe_get_image(example)
            gt_raw = example.get("ground_truth")
            if image is None or not isinstance(gt_raw, str):
                _maybe_log_scan_progress(f"cord/{split}", seen, len(samples), iter_started_at)
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
                words = sorted(
                    words,
                    key=lambda item: (
                        min(item["quad"].get(key, 0) for key in ["y1", "y2", "y3", "y4"]),
                        min(item["quad"].get(key, 0) for key in ["x1", "x2", "x3", "x4"]),
                    ),
                )
                text = _normalize_text(" ".join(word.get("text", "") for word in words))
                if len(text) < 2:
                    continue
                crop = _crop_image(image, _quad_to_bbox(words))
                if crop is None:
                    continue
                samples.append((crop, text))
                if len(samples) >= cap:
                    break
            _maybe_log_scan_progress(f"cord/{split}", seen, len(samples), iter_started_at)
            if len(samples) >= cap:
                break
        _record_count(summary, split, "cord", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, dataset_name, exc)
        _record_count(summary, split, "cord", 0)
        return []


def load_funsd_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    dataset_name = "nielsr/funsd"
    resolved_split = "test" if split == "validation" else split
    try:
        ds = _load_dataset_logged(dataset_name, resolved_split, cfg)
        samples: list[Sample] = []
        iter_started_at = time.perf_counter()
        for seen, example in enumerate(ds, start=1):
            image = _safe_get_image(example)
            if image is None:
                _maybe_log_scan_progress(f"funsd/{resolved_split}", seen, len(samples), iter_started_at)
                continue
            tokens: list[tuple[str, tuple[int, int, int, int]]] = []
            for word, box in zip(example.get("words", []), example.get("bboxes", [])):
                text = _normalize_text(str(word))
                if not text:
                    continue
                tokens.append((text, _resolve_bbox(box, image)))
            samples.extend(_extract_line_crops(image, tokens, max(0, cap - len(samples))))
            _maybe_log_scan_progress(f"funsd/{resolved_split}", seen, len(samples), iter_started_at)
            if len(samples) >= cap:
                break
        _record_count(summary, split, "funsd", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, dataset_name, exc)
        _record_count(summary, split, "funsd", 0)
        return []


def load_doclaynet_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    dataset_name = "docling-project/DocLayNet-v1.2"
    resolved_split = "validation" if split == "validation" else split
    try:
        ds = _load_dataset_logged(dataset_name, resolved_split, cfg)
        samples: list[Sample] = []
        iter_started_at = time.perf_counter()
        for seen, example in enumerate(ds, start=1):
            image = _safe_get_image(example)
            if image is None:
                _maybe_log_scan_progress(f"doclaynet/{resolved_split}", seen, len(samples), iter_started_at)
                continue
            for region_bbox, region_cells in zip(example.get("bboxes", []), example.get("pdf_cells", [])):
                if len(samples) >= cap:
                    break
                if not isinstance(region_cells, list) or not region_cells:
                    continue
                tokens: list[tuple[str, tuple[int, int, int, int]]] = []
                for cell in region_cells:
                    text = _normalize_text(str(cell.get("text", "")), preserve_newlines=False)
                    bbox = cell.get("bbox")
                    if not text or not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    tokens.append((text, _resolve_xywh_bbox(bbox, image)))
                if not tokens:
                    continue
                region = _resolve_xywh_bbox(region_bbox, image)
                if region[2] - region[0] < 8 or region[3] - region[1] < 8:
                    continue
                for sample in _extract_line_crops(image, tokens, max(0, cap - len(samples))):
                    samples.append(sample)
                    if len(samples) >= cap:
                        break
            _maybe_log_scan_progress(f"doclaynet/{resolved_split}", seen, len(samples), iter_started_at)
            if len(samples) >= cap:
                break
        _record_count(summary, split, "doclaynet", len(samples))
        return samples
    except Exception as exc:
        _record_warning(summary, dataset_name, exc)
        _record_count(summary, split, "doclaynet", 0)
        return []


def _download_file(url: str, target_path: Path) -> Path:
    if target_path.exists():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(target_path)
    return target_path


def _ensure_extracted(zip_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / ".complete"
    if marker.exists():
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    marker.touch()
    return extract_dir


def _build_file_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def _prepare_xfund_split(cache_dir: str | None, lang: str, split: str) -> tuple[Path, Path]:
    if lang not in _XFUND_LANGS:
        raise ValueError(f"Unsupported XFUN language: {lang}")
    resolved_split = "val" if split == "validation" else "train"
    root = Path(cache_dir or Path.home() / ".cache" / "ittamt_datasets") / "xfund" / lang / resolved_split
    json_path = root / f"{lang}.{resolved_split}.json"
    zip_path = root / f"{lang}.{resolved_split}.zip"
    images_dir = root / "images"
    _download_file(f"{_XFUND_BASE_URL}{lang}.{resolved_split}.json", json_path)
    _download_file(f"{_XFUND_BASE_URL}{lang}.{resolved_split}.zip", zip_path)
    _ensure_extracted(zip_path, images_dir)
    return json_path, images_dir


def load_xfund_split(split: str, cap: int, summary: dict[str, Any], cfg: DataConfig) -> list[Sample]:
    samples: list[Sample] = []
    for lang in cfg.xfund_languages:
        try:
            json_path, images_dir = _prepare_xfund_split(_cache_dir(cfg), lang, split)
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            image_index = _build_file_index(images_dir)
            for document in payload.get("documents", []):
                if len(samples) >= cap:
                    break
                image_name = document.get("img", {}).get("fname")
                image_path = image_index.get(image_name)
                if image_path is None:
                    continue
                image = Image.open(image_path).convert("RGB")
                for line in document.get("document", []):
                    if len(samples) >= cap:
                        break
                    text = _normalize_text(str(line.get("text", "")), preserve_newlines=False)
                    if len(text) < 2:
                        continue
                    boxes = [
                        _simplify_quad_bbox(word.get("box", []))
                        for word in line.get("words", [])
                        if isinstance(word, dict) and isinstance(word.get("box"), list) and len(word["box"]) >= 8
                    ]
                    if not boxes:
                        continue
                    crop = _crop_image(image, _union_bbox(boxes))
                    if crop is None:
                        continue
                    samples.append((crop, text))
                    if len(samples) >= cap:
                        break
            if len(samples) >= cap:
                break
        except Exception as exc:
            _record_warning(summary, f"XFUND[{lang}]", exc)
            continue
    _record_count(summary, split, "xfund", len(samples))
    return samples


def build_dataloaders(tokenizer: CharTokenizer, cfg: DataConfig):
    summary: dict[str, Any] = {"train": {}, "validation": {}, "warnings": []}
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    train_parts: list[Dataset[Any]] = []
    val_parts: list[Dataset[Any]] = []

    def _extend_with_stage(
        target: list[Sample],
        split: str,
        display_name: str,
        cap: int,
        loader: Any,
    ) -> None:
        _data_log(f"start {display_name} {split} (cap={cap})")
        started_at = time.perf_counter()
        loaded = loader(split, cap, summary, cfg)
        elapsed = time.perf_counter() - started_at
        target.extend(loaded)
        _data_log(f"done {display_name} {split}: +{len(loaded)} samples in {elapsed:.1f}s")

    if cfg.use_synthetic:
        _data_log(
            f"register lazy synthetic datasets train={cfg.synthetic_samples} validation={cfg.synthetic_val_samples}"
        )
        train_parts.append(SyntheticOCRDataset(cfg.synthetic_samples, tokenizer, cfg, seed=1337))
        val_parts.append(SyntheticOCRDataset(cfg.synthetic_val_samples, tokenizer, cfg, seed=7331))
        _record_count(summary, "train", "synthetic", cfg.synthetic_samples)
        _record_count(summary, "validation", "synthetic", cfg.synthetic_val_samples)

    if cfg.use_iam:
        _extend_with_stage(train_samples, "train", "iam", cfg.iam_train_cap, load_iam_split)
        _extend_with_stage(val_samples, "validation", "iam", cfg.iam_val_cap, load_iam_split)

    if cfg.use_iiit5k:
        _extend_with_stage(train_samples, "train", "iiit5k", cfg.iiit5k_train_cap, load_iiit5k_split)
        _extend_with_stage(val_samples, "validation", "iiit5k", cfg.iiit5k_val_cap, load_iiit5k_split)

    if cfg.use_textocr:
        _extend_with_stage(train_samples, "train", "textocr", cfg.textocr_train_cap, load_textocr_split)
        _extend_with_stage(val_samples, "validation", "textocr", cfg.textocr_val_cap, load_textocr_split)

    if cfg.use_sroie:
        _extend_with_stage(train_samples, "train", "sroie", cfg.sroie_train_cap, load_sroie_split)
        _extend_with_stage(val_samples, "validation", "sroie", cfg.sroie_val_cap, load_sroie_split)

    if cfg.use_cord:
        _extend_with_stage(train_samples, "train", "cord", cfg.cord_train_cap, load_cord_split)
        _extend_with_stage(val_samples, "validation", "cord", cfg.cord_val_cap, load_cord_split)

    if cfg.use_funsd:
        _extend_with_stage(train_samples, "train", "funsd", cfg.funsd_train_cap, load_funsd_split)
        _extend_with_stage(val_samples, "validation", "funsd", cfg.funsd_val_cap, load_funsd_split)

    if cfg.use_doclaynet:
        _extend_with_stage(train_samples, "train", "doclaynet", cfg.doclaynet_train_cap, load_doclaynet_split)
        _extend_with_stage(val_samples, "validation", "doclaynet", cfg.doclaynet_val_cap, load_doclaynet_split)

    if cfg.use_xfund:
        _extend_with_stage(train_samples, "train", "xfund", cfg.xfund_train_cap, load_xfund_split)
        _extend_with_stage(val_samples, "validation", "xfund", cfg.xfund_val_cap, load_xfund_split)

    _data_log(f"shuffle train/validation samples: train={len(train_samples)} validation={len(val_samples)}")
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    _data_log("constructing dataset objects")
    if train_samples:
        train_parts.append(OCRDataset(train_samples, tokenizer, cfg))
    if val_samples:
        val_parts.append(OCRDataset(val_samples, tokenizer, cfg))
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    collate_fn = lambda batch: _collate(batch, pad_id=tokenizer.blank_id, seq_pad_id=tokenizer.pad_id)
    loader_kwargs: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "collate_fn": collate_fn,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(2, cfg.prefetch_factor)

    _data_log(
        f"creating DataLoaders batch_size={cfg.batch_size} num_workers={cfg.num_workers} pin_memory={cfg.pin_memory}"
    )
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
    _data_log("dataloaders ready")
    return train_loader, val_loader, summary
