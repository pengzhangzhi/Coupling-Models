from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one_step_mnist import (  # noqa: E402
    IMAGE_SIZE,
    NUM_CLASSES,
    _gray_to_rgb,
    _to_uint8_0_255,
    build_mnist_fid,
    compute_fid_metric_value,
    ensure_dir,
)
from train_reward_model import RewardCNN, load_checkpoint as load_reward_checkpoint  # noqa: E402


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def load_mdm_training_args(checkpoint_path: Path, parser: argparse.ArgumentParser) -> argparse.Namespace:
    defaults = vars(parser.parse_args([]))
    payload = torch.load(checkpoint_path, map_location="cpu")
    merged = dict(defaults)
    merged.update(payload.get("args", {}))
    return argparse.Namespace(**merged)


def load_classifier_model(checkpoint_path: Path, device: torch.device) -> tuple[RewardCNN, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = payload.get("args", {})
    model = RewardCNN(
        num_classes=NUM_CLASSES,
        width=ckpt_args.get("model_width", 64),
        dropout=ckpt_args.get("dropout", 0.1),
    ).to(device)
    payload = load_reward_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    freeze_module(model)
    return model, payload


@torch.no_grad()
def classifier_alignment_metrics(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    target_probs = probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
    return {
        "accuracy": (preds == labels).float().mean().item(),
        "target_probability": target_probs.mean().item(),
        "target_log_probability": target_probs.log().mean().item(),
    }


def target_log_prob(classifier_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(classifier_logits, dim=-1).gather(1, labels[:, None]).squeeze(1)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_sample_grid(output_dir: Path, name: str, images: torch.Tensor, nrow: int = 8) -> None:
    ensure_dir(output_dir)
    save_image(images.detach().cpu(), output_dir / name, nrow=nrow)


def new_fid(data_dir: str, num_workers: int, device: torch.device, target_class: int | None):
    return build_mnist_fid(data_dir, num_workers, device, target_class=target_class)


def update_fid_with_images(fid, images: torch.Tensor) -> None:
    fid.update(_gray_to_rgb(_to_uint8_0_255(images)), real=False)


def fid_value(fid) -> float:
    return compute_fid_metric_value(fid)


def make_eval_labels(batch_size: int, target_class: int | None, device: torch.device, offset: int = 0) -> torch.Tensor:
    if target_class is not None:
        return torch.full((batch_size,), target_class, device=device, dtype=torch.long)
    base = torch.arange(NUM_CLASSES, device=device)
    repeats = (batch_size + offset + NUM_CLASSES - 1) // NUM_CLASSES
    return base.repeat(repeats)[offset : offset + batch_size]


def dataloader_workers(num_workers: int) -> int:
    return max(0, int(num_workers))
