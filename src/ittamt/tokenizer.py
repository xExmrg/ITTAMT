from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


DEFAULT_CHARS = list("\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?@#$%^&*()-_=+[]{}<>/\\|'\"`~")
SPECIAL_TOKENS = ["<blank>", "<pad>", "<bos>", "<eos>", "<mask>"]


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
        tokens = [self.bos_id] + self.encode(text) + [self.eos_id]
        if max_length is not None:
            if max_length <= 0:
                return []
            tokens = tokens[:max_length]
            if tokens and tokens[-1] != self.eos_id and len(tokens) == max_length:
                tokens[-1] = self.eos_id
        return tokens

    def batch_encode_sequence(self, texts: Iterable[str], max_length: int) -> tuple[list[list[int]], list[int]]:
        encoded = [self.encode_sequence(text, max_length=max_length) for text in texts]
        lengths = [len(seq) for seq in encoded]
        return encoded, lengths

    def decode_ctc(self, ids: list[int]) -> str:
        out = []
        prev = None
        for i in ids:
            if i == self.blank_id:
                prev = i
                continue
            if i != prev and 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok not in SPECIAL_TOKENS:
                    out.append(tok)
            prev = i
        return "".join(out)

    def decode_sequence(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in {self.pad_id, self.bos_id, self.blank_id, self.mask_id}:
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok not in SPECIAL_TOKENS:
                    out.append(tok)
        return "".join(out)

    def decode_sequence_batch(self, batch_ids: Iterable[list[int]]) -> list[str]:
        return [self.decode_sequence(ids) for ids in batch_ids]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _ensure_min_special_tokens(itos: list[str]) -> list[str]:
        """Backwards-compatible load for older tokenizer.json files.

        If a tokenizer is missing newer special tokens, insert them without reordering
        existing non-special tokens. We always keep <blank> at index 0 to preserve CTC
        semantics. If the format is unexpected, return it unchanged.
        """
        if not itos:
            return SPECIAL_TOKENS + DEFAULT_CHARS
        if itos[0] != "<blank>":
            return itos

        existing = set(itos)
        missing = [tok for tok in SPECIAL_TOKENS if tok not in existing]
        if not missing:
            return itos

        # Keep <blank> at 0; insert missing specials in the canonical SPECIAL_TOKENS order
        # after <blank>, but before any non-special tokens.
        out = itos[:]
        insert_at = 1
        for tok in SPECIAL_TOKENS[1:]:
            if tok in existing:
                # Advance past existing specials that appear near the front.
                insert_at = min(len(out), insert_at + 1)
                continue
            out.insert(insert_at, tok)
            insert_at += 1
        return out

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos = cls._ensure_min_special_tokens(list(data["itos"]))
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
