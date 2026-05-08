from __future__ import annotations

import re
from pathlib import Path

import torch

from utils.enhancer import EnhancerDatasetConfig
from utils.enhancer_models import FisherFlowEnhancerClassifier


FROZEN_CLASSIFIER_EMBED_DIM = 128


def upgrade_state_dict(state_dict: dict[str, torch.Tensor], prefixes: list[str] | None = None) -> dict[str, torch.Tensor]:
    if prefixes is None:
        prefixes = ["model."]
    pattern = re.compile("^" + "|".join(re.escape(prefix) for prefix in prefixes))
    return {pattern.sub("", name): param for name, param in state_dict.items()}


def build_enhancer_classifier(dataset: EnhancerDatasetConfig) -> FisherFlowEnhancerClassifier:
    return FisherFlowEnhancerClassifier(
        dim=dataset.vocab_size,
        k=dataset.seq_len,
        hidden=FROZEN_CLASSIFIER_EMBED_DIM,
        num_cls=dataset.num_classes,
        depth=dataset.classifier_depth,
        dropout=0.0,
        mode="",
        prior_pseudocount=2.0,
        cls_expanded_simplex=False,
        clean_data=True,
        classifier=True,
        classifier_free_guidance=False,
    )


class FrozenEnhancerClassifierEmbedder(torch.nn.Module):
    def __init__(self, dataset: EnhancerDatasetConfig, checkpoint_path: str | Path, device: torch.device):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Enhancer classifier checkpoint not found: {self.checkpoint_path}")

        self.classifier = build_enhancer_classifier(dataset)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload["state_dict"] if "state_dict" in payload else payload
        self.classifier.load_state_dict(upgrade_state_dict(state_dict, prefixes=["model."]))
        self.classifier.to(device).eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return FROZEN_CLASSIFIER_EMBED_DIM

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, embeddings = self.classifier(tokens.to(self.device), t=None, return_embedding=True)
        return embeddings.float()

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(tokens.to(self.device), t=None).float()
