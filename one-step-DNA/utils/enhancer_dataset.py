from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.enhancer import EnhancerDatasetConfig


class EnhancerTransportDatasetView(Dataset):
    def __init__(self, data_dir: str | Path, dataset: EnhancerDatasetConfig, split: str):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split '{split}'.")
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.split = split

        payload = pickle.load(open(self.data_dir / dataset.file_name, "rb"))
        self.seq_onehot = torch.from_numpy(payload[f"{split}_data"]).float()
        self.seq_tokens = self.seq_onehot.argmax(dim=-1).long()
        self.labels = torch.from_numpy(payload[f"y_{split}"]).argmax(dim=-1).long()

        if self.seq_onehot.ndim != 3 or self.seq_onehot.shape[1] != dataset.seq_len or self.seq_onehot.shape[2] != dataset.vocab_size:
            raise ValueError(
                f"Unexpected sequence tensor shape for {dataset.name}/{split}: {tuple(self.seq_onehot.shape)}"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"Unexpected label tensor shape for {dataset.name}/{split}: {tuple(self.labels.shape)}")

    def __len__(self) -> int:
        return self.seq_tokens.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.seq_onehot[index], self.seq_tokens[index], self.labels[index]
