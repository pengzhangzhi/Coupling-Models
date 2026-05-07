from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import torch


def _normalize_ratio(value: float | str) -> Fraction:
    if isinstance(value, str):
        value = value.strip()
        if "/" in value:
            num, den = value.split("/", 1)
            return Fraction(int(num), int(den))
        return Fraction(value)
    return Fraction(value).limit_denominator()


@dataclass(frozen=True)
class EnhancerDatasetConfig:
    name: str
    file_name: str
    num_classes: int
    classifier_depth: int
    classifier_relpath: str
    seq_len: int = 500
    vocab_size: int = 4


ENHANCER_DATASETS = {
    "fb": EnhancerDatasetConfig(
        name="fb",
        file_name="DeepFlyBrain_data.pkl",
        num_classes=81,
        classifier_depth=1,
        classifier_relpath="workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt",
    ),
    "mel": EnhancerDatasetConfig(
        name="mel",
        file_name="DeepMEL2_data.pkl",
        num_classes=47,
        classifier_depth=4,
        classifier_relpath="workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt",
    ),
}


def get_enhancer_dataset_config(name: str) -> EnhancerDatasetConfig:
    key = name.strip().lower()
    aliases = {
        "fly": "fb",
        "flybrain": "fb",
        "deepflybrain": "fb",
        "dna": "fb",
        "mel2": "mel",
        "melanoma": "mel",
        "deepmel2": "mel",
    }
    key = aliases.get(key, key)
    if key not in ENHANCER_DATASETS:
        raise ValueError(f"Unknown enhancer dataset '{name}'. Expected one of: {sorted(ENHANCER_DATASETS)}")
    return ENHANCER_DATASETS[key]


@dataclass(frozen=True)
class EnhancerCompressionSpec:
    seq_len: int
    compression_ratio: float
    latent_length: int

    @classmethod
    def from_ratio(cls, seq_len: int, compression_ratio: float | str) -> "EnhancerCompressionSpec":
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        ratio_fraction = _normalize_ratio(compression_ratio)
        if ratio_fraction <= 0 or ratio_fraction > 1:
            raise ValueError("compression_ratio must be in (0, 1].")
        latent_length_fraction = ratio_fraction * seq_len
        if latent_length_fraction.denominator != 1:
            raise ValueError(
                f"compression_ratio={float(ratio_fraction)} does not produce an integer latent length for seq_len={seq_len}."
            )
        latent_length = latent_length_fraction.numerator
        if latent_length < 1:
            raise ValueError("compression_ratio produced an empty latent sequence.")
        return cls(
            seq_len=seq_len,
            compression_ratio=float(ratio_fraction),
            latent_length=latent_length,
        )


def default_classifier_checkpoint(data_dir: str | Path, dataset: EnhancerDatasetConfig) -> Path:
    data_root = Path(data_dir).resolve().parent
    return data_root / dataset.classifier_relpath


def decode_tokens(tokens: torch.Tensor) -> list[str]:
    alphabet = "ACGT"
    return ["".join(alphabet[idx] for idx in row.tolist()) for row in tokens]


def balanced_divisor_pair(n: int) -> tuple[int, int]:
    root = math.isqrt(n)
    for width in range(root, 0, -1):
        if n % width == 0:
            height = n // width
            if height < width:
                height, width = width, height
            return height, width
    return n, 1
