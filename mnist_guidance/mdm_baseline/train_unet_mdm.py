from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one_step_mnist import ensure_dir, get_device, make_scheduler, set_seed  # noqa: E402
from mdm_baseline.model import MASK_TOKEN  # noqa: E402
from mdm_baseline.unet_model import (  # noqa: E402
    UNET_IMAGE_SIZE,
    count_parameters,
    load_unet_checkpoint,
    make_unet_mdm_from_args,
    save_unet_checkpoint,
)


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "checkpoints"


def last_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path)
    return checkpoint_dir(args) / "last.pt"


def best_checkpoint_path(args: argparse.Namespace) -> Path:
    return checkpoint_dir(args) / "best.pt"


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_path is not None:
        return Path(args.metrics_path)
    return Path(args.output_dir) / "metrics.jsonl"


def append_metrics(args: argparse.Namespace, record: dict) -> None:
    path = metrics_path(args)
    ensure_dir(path.parent)
    payload = {
        "seed": args.seed,
        "output_dir": str(Path(args.output_dir).resolve()),
        "objective": args.objective,
        "base_channels": args.base_channels,
        "channel_mult": args.channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "attention_resolutions": args.attention_resolutions,
        "cfg_drop_prob": args.cfg_drop_prob,
    }
    payload.update(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class BinaryMNIST32(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, download: bool = True, random_flip: bool = False):
        transform_parts = [
            transforms.Resize(UNET_IMAGE_SIZE),
            transforms.CenterCrop(UNET_IMAGE_SIZE),
        ]
        if random_flip:
            transform_parts.append(transforms.RandomHorizontalFlip())
        transform_parts.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x >= 0.5).float()),
            ]
        )
        self.dataset = datasets.MNIST(root=root, train=train, download=download, transform=transforms.Compose(transform_parts))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        return image, image.long().flatten(), torch.tensor(label, dtype=torch.long)


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_set = BinaryMNIST32(args.data_dir, train=True, download=True, random_flip=args.random_flip)
    test_set = BinaryMNIST32(args.data_dir, train=False, download=True, random_flip=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader


def md4_masking_schedule(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.cos((1 - t) * torch.pi / 2)


def md4_loss_weight(t: torch.Tensor) -> torch.Tensor:
    return torch.pi * torch.tan((1 - t).clamp_max(1 - 1e-4) * torch.pi / 2) / 2


def sample_times(batch_size: int, device: torch.device, antithetic: bool) -> torch.Tensor:
    if not antithetic:
        return torch.rand(batch_size, device=device).clamp(1e-4, 1 - 1e-4)
    offset = torch.rand((), device=device)
    t = (offset + torch.arange(batch_size, device=device).float() / batch_size).fmod(1.0)
    return t.clamp(1e-4, 1 - 1e-4)


def corrupt_tokens(clean: torch.Tensor, t: torch.Tensor, objective: str) -> tuple[torch.Tensor, torch.Tensor]:
    if objective == "md4":
        mask_prob = 1 - md4_masking_schedule(t)
    elif objective == "uniform":
        mask_prob = t
    else:
        raise ValueError(f"Unknown objective: {objective}")
    mask = torch.rand_like(clean.float()) < mask_prob[:, None]
    empty = ~mask.any(dim=1)
    if empty.any():
        forced = torch.randint(0, clean.shape[1], (int(empty.sum().item()),), device=clean.device)
        mask[empty, forced] = True
    tokens = clean.masked_fill(mask, MASK_TOKEN)
    return tokens, mask


def masked_bce_loss(logits: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, objective: str) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, clean.float(), reduction="none")
    if objective == "md4":
        loss = loss * md4_loss_weight(t)[:, None]
    return loss[mask].mean()


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for _, clean, labels in loader:
        clean = clean.to(device)
        labels = labels.to(device)
        t = sample_times(clean.shape[0], device, args.antithetic_time)
        tokens, mask = corrupt_tokens(clean, t, args.objective)
        logits = model(tokens, t, labels)
        loss = masked_bce_loss(logits, clean, mask, t, args.objective)
        total_loss += loss.item()
        total_batches += 1
    model.train(was_training)
    return total_loss / max(1, total_batches)


def bernoulli_log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = logits / temperature
    return torch.stack([F.logsigmoid(-scaled), F.logsigmoid(scaled)], dim=-1)


@torch.no_grad()
def sample_unet_mdm(
    model,
    labels: torch.Tensor,
    steps: int,
    temperature: float,
    argmax: bool,
    sampler: str,
) -> torch.Tensor:
    tokens = torch.full((labels.shape[0], model.num_pixels), MASK_TOKEN, device=labels.device, dtype=torch.long)
    if sampler == "confidence":
        for step in range(steps):
            masked = tokens == MASK_TOKEN
            remaining = masked.sum(dim=1)
            if int(remaining.max().item()) == 0:
                break
            t = (remaining.float() / model.num_pixels).clamp(1e-4, 1 - 1e-4)
            logits = model(tokens, t, labels)
            log_probs = bernoulli_log_probs(logits, temperature)
            probs = log_probs.exp()
            samples = log_probs.argmax(dim=-1) if argmax else torch.distributions.Categorical(probs=probs).sample()
            confidence = probs.max(dim=-1).values.masked_fill(~masked, -1.0)
            reveal_count = int(torch.ceil(remaining.float() / (steps - step)).max().item())
            _, indices = confidence.topk(k=min(reveal_count, int(remaining.max().item())), dim=1)
            reveal = torch.zeros_like(masked)
            reveal.scatter_(1, indices, True)
            reveal &= masked
            tokens = torch.where(reveal, samples, tokens)
    elif sampler == "md4":
        t_grid = torch.linspace(0, 1, steps + 1, device=labels.device)
        for i in range(steps, 1, -1):
            masked = tokens == MASK_TOKEN
            if int(masked.sum().item()) == 0:
                break
            ti = torch.full((labels.shape[0],), t_grid[i].item(), device=labels.device)
            si = torch.full((labels.shape[0],), t_grid[i - 1].item(), device=labels.device)
            alpha_t = md4_masking_schedule(ti)
            alpha_s = md4_masking_schedule(si)
            unmask_prob = ((alpha_s - alpha_t) / (1 - alpha_t).clamp_min(1e-6)).clamp(0, 1)
            logits = model(tokens, ti, labels)
            log_probs = bernoulli_log_probs(logits, temperature)
            probs = log_probs.exp()
            samples = log_probs.argmax(dim=-1) if argmax else torch.distributions.Categorical(probs=probs).sample()
            reveal = (torch.rand_like(tokens.float()) < unmask_prob[:, None]) & masked
            tokens = torch.where(reveal, samples, tokens)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    tokens = tokens.masked_fill(tokens == MASK_TOKEN, 0)
    return tokens.float().view(-1, 1, UNET_IMAGE_SIZE, UNET_IMAGE_SIZE)


@torch.no_grad()
def save_samples(model, args: argparse.Namespace, epoch: int, device: torch.device) -> None:
    labels = torch.arange(10, device=device).repeat_interleave(10)
    images = sample_unet_mdm(
        model,
        labels,
        steps=args.sample_steps,
        temperature=args.sample_temperature,
        argmax=args.sample_argmax,
        sampler=args.sample_sampler,
    )
    suffix = "final" if epoch < 0 else f"epoch_{epoch:03d}"
    path = Path(args.output_dir) / f"samples_{suffix}.png"
    save_image(images.cpu(), path, nrow=10)
    print(f"[samples] saved {path}")


def train(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output_dir))
    set_seed(args.seed)
    device = get_device()
    train_loader, test_loader = make_dataloaders(args)
    model = make_unet_mdm_from_args(args, device)
    param_count = count_parameters(model)
    print(f"[model] parameters={param_count:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, len(train_loader), args.epochs)

    global_step = 0
    start_epoch = 1
    best_eval_loss = float("inf")
    if args.resume:
        payload = load_unet_checkpoint(last_checkpoint_path(args), model, optimizer, device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        best_eval_loss = float(payload.get("best_eval_loss", best_eval_loss))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_steps = 0
        for step, (_, clean, labels) in enumerate(train_loader, start=1):
            clean = clean.to(device)
            labels = labels.to(device)
            t = sample_times(clean.shape[0], device, args.antithetic_time)
            tokens, mask = corrupt_tokens(clean, t, args.objective)
            drop_mask = torch.rand(labels.shape[0], device=device) < args.cfg_drop_prob
            logits = model(tokens, t, labels, drop_mask=drop_mask)
            loss = masked_bce_loss(logits, clean, mask, t, args.objective)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pred = (logits[mask] >= 0).long()
                acc = (pred == clean[mask]).float().mean().item()
            total_loss += loss.item()
            total_acc += acc
            total_steps += 1

            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch:03d} step={step:04d} "
                    f"loss={loss.item():.4f} acc={acc:.4f} grad={float(grad_norm):.4f}"
                )
            global_step += 1

        train_loss = total_loss / max(1, total_steps)
        train_acc = total_acc / max(1, total_steps)
        eval_loss = evaluate_loss(model, test_loader, device, args)
        print(f"[epoch] epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} eval_loss={eval_loss:.4f}")
        append_metrics(
            args,
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "parameters": param_count,
            },
        )

        save_unet_checkpoint(
            last_checkpoint_path(args),
            model,
            optimizer,
            args,
            epoch,
            global_step,
            extra={"best_eval_loss": best_eval_loss, "parameters": param_count},
        )
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_unet_checkpoint(
                best_checkpoint_path(args),
                model,
                optimizer,
                args,
                epoch,
                global_step,
                extra={"best_eval_loss": best_eval_loss, "parameters": param_count},
            )
        if args.sample_every > 0 and (epoch % args.sample_every == 0 or epoch == args.epochs):
            save_samples(model, args, epoch, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 32x32 U-Net MDM baseline on binary MNIST.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/unet_mdm_baseline")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--objective", choices=("md4", "uniform"), default="md4")
    parser.add_argument("--antithetic-time", action="store_true", default=True)
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-mult", type=str, default="1,2,2,4")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", type=str, default="8,16")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=256)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-argmax", action="store_true")
    parser.add_argument("--sample-sampler", choices=("confidence", "md4"), default="md4")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if not 0 <= args.cfg_drop_prob <= 1:
        raise ValueError("--cfg-drop-prob must be in [0, 1]")
    if args.base_channels % args.num_heads != 0:
        raise ValueError("--base-channels must be divisible by --num-heads")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()

