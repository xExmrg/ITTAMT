from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


DEFAULT_CHARS = list("\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?@#$%^&*()-_=+[]{}<>/\\|'\"`~")
SPECIAL_TOKENS = ["<blank>", "<pad>", "<bos>", "<eos>", "<mask>"]


def _ensure_min_special_tokens(itos: list[str]) -> list[str]:
    if not itos:
        return SPECIAL_TOKENS + DEFAULT_CHARS

    normalized = list(itos)
    if normalized[0] != "<blank>":
        normalized = ["<blank>"] + [token for token in normalized if token != "<blank>"]

    insert_at = 1
    for token in SPECIAL_TOKENS[1:]:
        if token not in normalized:
            normalized.insert(insert_at, token)
            insert_at += 1
        else:
            insert_at = normalized.index(token) + 1
    return normalized


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build_default(cls) -> "CharTokenizer":
        itos = SPECIAL_TOKENS + DEFAULT_CHARS
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def blank_id(self) -> int:
        return self.stoi["<blank>"]

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def mask_id(self) -> int:
        return self.stoi["<mask>"]

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        return ids

    def encode_sequence(self, text: str, max_length: int | None = None) -> list[int]:
        ids = [self.bos_id, *self.encode(text), self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
            if ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def batch_encode_sequence(self, texts: Iterable[str], max_length: int | None = None) -> list[list[int]]:
        return [self.encode_sequence(text, max_length=max_length) for text in texts]

    def decode_ctc(self, ids: list[int]) -> str:
        out = []
        prev = None
        for i in ids:
            if i == self.blank_id:
                prev = i
                continue
            if i != prev and i < len(self.itos):
                tok = self.itos[i]
                if tok not in SPECIAL_TOKENS:
                    out.append(tok)
            prev = i
        return "".join(out)

    def decode_sequence(self, ids: list[int]) -> str:
        out = []
        for token_id in ids:
            if token_id >= len(self.itos):
                continue
            token = self.itos[token_id]
            if token == "<eos>":
                break
            if token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return "".join(out)

    def decode_sequence_batch(self, batch_ids: Iterable[list[int]]) -> list[str]:
        return [self.decode_sequence(ids) for ids in batch_ids]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos = _ensure_min_special_tokens(data["itos"])
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
