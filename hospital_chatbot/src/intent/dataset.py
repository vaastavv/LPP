"""Torch dataset + label mapping for intent fine-tuning."""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import INTENTS
from src.text import normalize


def label_maps(labels: list[str] = INTENTS) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {l: i for i, l in enumerate(labels)}
    return label2id, {i: l for l, i in label2id.items()}


class IntentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: dict[str, int], max_length: int):
        texts = [normalize(t) for t in df["text"]]
        self.enc = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [label2id[l] for l in df["intent"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def collate_fn(tokenizer):
    def _collate(batch: list[dict]) -> dict:
        labels = torch.tensor([b.pop("labels") for b in batch])
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return _collate
