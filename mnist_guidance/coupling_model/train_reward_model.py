import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from one_step_mnist import IMAGE_SIZE, NUM_CLASSES, ensure_dir, get_device, set_seed

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def log_wandb(data: dict) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(data)


def init_wandb(args: argparse.Namespace) -> None:
    if args.no_wandb or wandb is None:
        return
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        dir=args.output_dir,
    )


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_path is not None:
        return Path(args.metrics_path)
    return Path(args.output_dir) / "metrics.jsonl"


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "checkpoints"


def last_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path)
    return checkpoint_dir(args) / "last.pt"


def best_checkpoint_path(args: argparse.Namespace) -> Path:
    return checkpoint_dir(args) / "best.pt"


def append_metrics_record(
    args: argparse.Namespace,
    event: str,
    epoch: int | None,
    **metrics: float | int | str | None,
) -> None:
    path = metrics_path(args)
    ensure_dir(path.parent)
    record = {
        "event": event,
        "epoch": epoch,
        "seed": args.seed,
        "output_dir": str(Path(args.output_dir).resolve()),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gray_mix_prob": args.gray_mix_prob,
        "binary_mix_prob": args.binary_mix_prob,
        "soft_mix_prob": args.soft_mix_prob,
        "soft_noise_std": args.soft_noise_std,
        "soft_interp_alpha": args.soft_interp_alpha,
        "soft_midgray_alpha": args.soft_midgray_alpha,
    }
    record.update(metrics)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


class RewardCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
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


def binarize_images(images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (images >= threshold).to(images.dtype)


def soften_images(
    images: torch.Tensor,
    noise_std: float,
    interp_alpha: float,
    midgray_alpha: float,
) -> torch.Tensor:
    soft = images
    if interp_alpha > 0:
        mix = torch.rand_like(soft)
        alpha = torch.rand(soft.shape[0], 1, 1, 1, device=soft.device, dtype=soft.dtype) * interp_alpha
        soft = (1 - alpha) * soft + alpha * mix
    if midgray_alpha > 0:
        alpha_mid = torch.rand(soft.shape[0], 1, 1, 1, device=soft.device, dtype=soft.dtype) * midgray_alpha
        soft = (1 - alpha_mid) * soft + alpha_mid * 0.5
    if noise_std > 0:
        std = torch.rand(soft.shape[0], 1, 1, 1, device=soft.device, dtype=soft.dtype) * noise_std
        soft = soft + torch.randn_like(soft) * std
    return soft.clamp_(0.0, 1.0)


def make_mixed_training_inputs(images: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    binary = binarize_images(images)
    # Start from either grayscale or hard binary, then optionally soften.
    soft_base = torch.where(
        (torch.rand(images.shape[0], 1, 1, 1, device=images.device) < 0.5),
        images,
        binary,
    )
    soft = soften_images(
        soft_base,
        noise_std=args.soft_noise_std,
        interp_alpha=args.soft_interp_alpha,
        midgray_alpha=args.soft_midgray_alpha,
    )

    probs = torch.rand(images.shape[0], device=images.device)
    gray_cutoff = args.gray_mix_prob
    binary_cutoff = gray_cutoff + args.binary_mix_prob

    view = soft.clone()
    gray_mask = probs < gray_cutoff
    binary_mask = (probs >= gray_cutoff) & (probs < binary_cutoff)

    if gray_mask.any():
        view[gray_mask] = images[gray_mask]
    if binary_mask.any():
        view[binary_mask] = binary[binary_mask]
    return view


def make_soft_eval_inputs(images: torch.Tensor, alpha: float) -> torch.Tensor:
    binary = binarize_images(images)
    return ((1 - alpha) * binary + alpha * 0.5).clamp(0.0, 1.0)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate_split(
    model: RewardCNN,
    loader: DataLoader,
    device: torch.device,
    split: str,
    soft_eval_alpha: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    total_conf = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if split == "gray":
            inputs = images
        elif split == "binary":
            inputs = binarize_images(images)
        elif split == "soft":
            inputs = make_soft_eval_inputs(images, alpha=soft_eval_alpha)
        else:
            raise ValueError(f"Unknown eval split: {split}")

        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        total_acc += (logits.argmax(dim=-1) == labels).float().sum().item()
        total_conf += probs.max(dim=-1).values.sum().item()
        total_count += labels.shape[0]

    model.train(was_training)
    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_acc / max(1, total_count),
        "confidence": total_conf / max(1, total_count),
    }


def save_checkpoint(
    path: Path,
    model: RewardCNN,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_mean_acc: float,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_mean_acc": best_mean_acc,
        "args": vars(args),
    }
    torch.save(payload, path)
    print(f"[checkpoint] saved {path}")


def load_checkpoint(
    path: Path,
    model: RewardCNN,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    print(f"[checkpoint] loaded {path}")
    return payload


def train(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output_dir))
    set_seed(args.seed)
    device = get_device()
    train_loader, test_loader = make_dataloaders(args)
    model = RewardCNN(width=args.model_width, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    global_step = 0
    best_mean_acc = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_conf = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            inputs = make_mixed_training_inputs(images, args)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_acc = accuracy_from_logits(logits, labels)
            batch_conf = probs.max(dim=-1).values.mean().item()
            epoch_loss += loss.item()
            epoch_acc += batch_acc
            epoch_conf += batch_conf

            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch:03d} step={step:04d} "
                    f"loss={loss.item():.4f} acc={batch_acc:.4f} conf={batch_conf:.4f}"
                )
                log_wandb(
                    {
                        "train/loss": loss.item(),
                        "train/acc": batch_acc,
                        "train/confidence": batch_conf,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }
                )
            global_step += 1

        num_steps = len(train_loader)
        avg_train_loss = epoch_loss / max(1, num_steps)
        avg_train_acc = epoch_acc / max(1, num_steps)
        avg_train_conf = epoch_conf / max(1, num_steps)
        print(
            f"[epoch] epoch={epoch:03d} train_loss={avg_train_loss:.4f} "
            f"train_acc={avg_train_acc:.4f} train_conf={avg_train_conf:.4f}"
        )
        append_metrics_record(
            args,
            event="train_epoch",
            epoch=epoch,
            train_loss=avg_train_loss,
            train_acc=avg_train_acc,
            train_confidence=avg_train_conf,
        )
        log_wandb(
            {
                "epoch/train_loss": avg_train_loss,
                "epoch/train_acc": avg_train_acc,
                "epoch/train_confidence": avg_train_conf,
                "epoch/epoch": epoch,
            }
        )

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue

        eval_gray = evaluate_split(model, test_loader, device, split="gray", soft_eval_alpha=args.soft_eval_alpha)
        eval_binary = evaluate_split(model, test_loader, device, split="binary", soft_eval_alpha=args.soft_eval_alpha)
        eval_soft = evaluate_split(model, test_loader, device, split="soft", soft_eval_alpha=args.soft_eval_alpha)
        mean_acc = (eval_gray["accuracy"] + eval_binary["accuracy"] + eval_soft["accuracy"]) / 3.0

        print(
            f"[eval] epoch={epoch:03d} "
            f"gray_acc={eval_gray['accuracy']:.4f} "
            f"binary_acc={eval_binary['accuracy']:.4f} "
            f"soft_acc={eval_soft['accuracy']:.4f} "
            f"mean_acc={mean_acc:.4f}"
        )
        append_metrics_record(
            args,
            event="eval_epoch",
            epoch=epoch,
            eval_gray_loss=eval_gray["loss"],
            eval_gray_acc=eval_gray["accuracy"],
            eval_binary_loss=eval_binary["loss"],
            eval_binary_acc=eval_binary["accuracy"],
            eval_soft_loss=eval_soft["loss"],
            eval_soft_acc=eval_soft["accuracy"],
            eval_mean_acc=mean_acc,
        )
        log_wandb(
            {
                "eval/gray_loss": eval_gray["loss"],
                "eval/gray_acc": eval_gray["accuracy"],
                "eval/binary_loss": eval_binary["loss"],
                "eval/binary_acc": eval_binary["accuracy"],
                "eval/soft_loss": eval_soft["loss"],
                "eval/soft_acc": eval_soft["accuracy"],
                "eval/mean_acc": mean_acc,
                "eval/epoch": epoch,
            }
        )

        save_checkpoint(last_checkpoint_path(args), model, optimizer, args, epoch, global_step, best_mean_acc)
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            save_checkpoint(best_checkpoint_path(args), model, optimizer, args, epoch, global_step, best_mean_acc)

    save_checkpoint(last_checkpoint_path(args), model, optimizer, args, args.epochs, global_step, best_mean_acc)


@torch.no_grad()
def evaluate_only(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output_dir))
    set_seed(args.seed)
    device = get_device()
    _, test_loader = make_dataloaders(args)
    model = RewardCNN(width=args.model_width, dropout=args.dropout).to(device)
    payload = load_checkpoint(last_checkpoint_path(args), model, optimizer=None, device=device)
    epoch = payload.get("epoch")

    eval_gray = evaluate_split(model, test_loader, device, split="gray", soft_eval_alpha=args.soft_eval_alpha)
    eval_binary = evaluate_split(model, test_loader, device, split="binary", soft_eval_alpha=args.soft_eval_alpha)
    eval_soft = evaluate_split(model, test_loader, device, split="soft", soft_eval_alpha=args.soft_eval_alpha)
    mean_acc = (eval_gray["accuracy"] + eval_binary["accuracy"] + eval_soft["accuracy"]) / 3.0

    print(
        f"[eval-only] epoch={epoch} "
        f"gray_acc={eval_gray['accuracy']:.4f} "
        f"binary_acc={eval_binary['accuracy']:.4f} "
        f"soft_acc={eval_soft['accuracy']:.4f} "
        f"mean_acc={mean_acc:.4f}"
    )
    append_metrics_record(
        args,
        event="eval_only",
        epoch=epoch,
        eval_gray_loss=eval_gray["loss"],
        eval_gray_acc=eval_gray["accuracy"],
        eval_binary_loss=eval_binary["loss"],
        eval_binary_acc=eval_binary["accuracy"],
        eval_soft_loss=eval_soft["loss"],
        eval_soft_acc=eval_soft["accuracy"],
        eval_mean_acc=mean_acc,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an MNIST reward model on mixed hard, grayscale, and soft inputs.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/reward_model")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gray-mix-prob", type=float, default=0.25)
    parser.add_argument("--binary-mix-prob", type=float, default=0.35)
    parser.add_argument("--soft-mix-prob", type=float, default=0.40)
    parser.add_argument("--soft-noise-std", type=float, default=0.12)
    parser.add_argument("--soft-interp-alpha", type=float, default=0.20)
    parser.add_argument("--soft-midgray-alpha", type=float, default=0.10)
    parser.add_argument("--soft-eval-alpha", type=float, default=0.15)
    parser.add_argument("--wandb-project", type=str, default="reward-model-mnist")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    mix_total = args.gray_mix_prob + args.binary_mix_prob + args.soft_mix_prob
    if abs(mix_total - 1.0) > 1e-6:
        raise ValueError(
            "Training mixture probabilities must sum to 1.0, got "
            f"{mix_total:.6f} from gray={args.gray_mix_prob}, "
            f"binary={args.binary_mix_prob}, soft={args.soft_mix_prob}."
        )
    if args.soft_noise_std < 0 or args.soft_interp_alpha < 0 or args.soft_midgray_alpha < 0 or args.soft_eval_alpha < 0:
        raise ValueError("Softening parameters must be non-negative.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    init_wandb(args)
    try:
        if args.eval_only:
            evaluate_only(args)
        else:
            train(args)
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
