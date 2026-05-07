from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F

from utils.enhancer import EnhancerDatasetConfig
from utils.enhancer_external import FrozenEnhancerClassifierEmbedder


def get_wasserstein_dist(embeds1: np.ndarray, embeds2: np.ndarray) -> float:
    if np.isnan(embeds1).any() or np.isnan(embeds2).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float("nan")
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


class EnhancerFBDEvaluator(torch.nn.Module):
    def __init__(self, dataset: EnhancerDatasetConfig, checkpoint_path: str | Path, device: torch.device):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.embedder = FrozenEnhancerClassifierEmbedder(dataset, checkpoint_path, device)
        self.cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def embeddings(self, tokens: torch.Tensor, cache_key: str | None = None) -> np.ndarray:
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        embeddings = self.embedder(tokens)
        value = embeddings.detach().cpu().float().numpy()
        if cache_key is not None:
            self.cache[cache_key] = value
        return value

    @torch.inference_mode()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedder.logits(tokens)

    @torch.inference_mode()
    def target_metrics(self, tokens: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        logits = self.logits(tokens)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        labels = labels.to(self.device)
        acc = (predictions == labels).float().mean().item()
        target_prob = probs.gather(dim=-1, index=labels[:, None]).mean().item()
        return acc, target_prob

    @torch.inference_mode()
    def fbd(self, generated: torch.Tensor, real: torch.Tensor, cache_key: str | None = None) -> float:
        gen_embeddings = self.embeddings(generated)
        real_embeddings = self.embeddings(real, cache_key=cache_key)
        return get_wasserstein_dist(gen_embeddings, real_embeddings)
