from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_CHARS = list("\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?@#$%^&*()-_=+[]{}<>/\\|'\"`~")
SPECIAL_TOKENS = ["<blank>", "<pad>"]


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

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        return ids

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

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos = data["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
