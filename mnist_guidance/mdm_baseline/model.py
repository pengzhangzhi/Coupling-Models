from __future__ import annotations

import torch

from one_step_mnist import IMAGE_SIZE


ZERO_TOKEN = 0
ONE_TOKEN = 1
MASK_TOKEN = 2
VOCAB_SIZE = 3


def tokens_to_images(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.float().view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


def binary_sequences_from_images(images: torch.Tensor) -> torch.Tensor:
    return (images >= 0.5).long().flatten(1)
