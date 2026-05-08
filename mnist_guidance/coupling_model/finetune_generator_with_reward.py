import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from eval_reward_guidance import (
    binary_concrete_samples,
    classifier_alignment_metrics,
    load_classifier_model,
    load_generator_training_args,
    straight_through_relaxed,
    target_log_prob,
)
from one_step_mnist import (
    FLOW_TOKENS,
    IMAGE_SIZE,
    NUM_CLASSES,
    _gray_to_rgb,
    _to_uint8_0_255,
    build_mnist_fid,
    compute_fid_metric_value,
    ensure_dir,
    get_device,
    is_cond_mode,
    load_checkpoint as load_generator_checkpoint,
    make_models,
    set_seed,
)
from train_reward_model import RewardCNN


FINETUNE_METHODS = ("ft-soft", "ft-relaxed")


def parse_target_class(value: str) -> int | None:
    if value.lower() in {"none", "balanced", "all"}:
        return None
    parsed = int(value)
    if parsed < 0 or parsed >= NUM_CLASSES:
        raise argparse.ArgumentTypeError(f"target class must be in [0, {NUM_CLASSES - 1}] or none")
    return parsed


def make_eval_labels(batch_size: int, target_class: int | None, device: torch.device, offset: int = 0) -> torch.Tensor:
    if target_class is not None:
        return torch.full((batch_size,), target_class, device=device, dtype=torch.long)
    base = torch.arange(NUM_CLASSES, device=device)
    repeats = (batch_size + offset + NUM_CLASSES - 1) // NUM_CLASSES
    return base.repeat(repeats)[offset : offset + batch_size]


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def reward_inputs_from_logits(logits: torch.Tensor, method: str, args: argparse.Namespace) -> torch.Tensor:
    if method == "ft-soft":
        samples = torch.sigmoid(logits)
    elif method == "ft-relaxed":
        if args.relaxed_method == "binary-concrete":
            samples = binary_concrete_samples(logits, temperature=args.relaxed_temperature)
        elif args.relaxed_method == "st":
            samples = straight_through_relaxed(logits)
        else:
            raise ValueError(f"Unknown relaxed method: {args.relaxed_method}")
    else:
        raise ValueError(f"Unknown fine-tuning method: {method}")
    return samples.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


def save_finetuned_checkpoint(
    path: Path,
    vae: nn.Module,
    flow: nn.Module,
    generator: nn.Module,
    gen_args: argparse.Namespace,
    finetune_args: argparse.Namespace,
    method: str,
    global_step: int,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "vae": vae.state_dict(),
        "flow": flow.state_dict(),
        "generator": generator.state_dict(),
        "epoch": 0,
        "global_step": global_step,
        "args": vars(gen_args),
        "finetune_args": vars(finetune_args),
        "finetune_method": method,
    }
    torch.save(payload, path)
    print(f"[checkpoint] saved {path}")


def train_one_method(
    method: str,
    generator: nn.Module,
    anchor_generator: nn.Module,
    reward_model: RewardCNN,
    gen_args: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    generator.train()
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    logs: list[dict[str, float | int | str]] = []

    for step in range(1, args.steps + 1):
        labels = make_eval_labels(args.batch_size, args.target_class, device, offset=((step - 1) * args.batch_size) % NUM_CLASSES)
        z_tokens = torch.randn(args.batch_size, FLOW_TOKENS, gen_args.latent_channels, device=device)
        logits = generator(z_tokens, labels)
        with torch.no_grad():
            anchor_logits = anchor_generator(z_tokens, labels)

        reward_inputs = reward_inputs_from_logits(logits, method, args)
        reward_logits = reward_model(reward_inputs)
        reward_log_prob = target_log_prob(reward_logits, labels)
        anchor_loss = F.mse_loss(logits, anchor_logits)
        reward_loss = -reward_log_prob.mean()
        total_loss = args.lambda_reward * reward_loss + args.lambda_anchor * anchor_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "method": method,
                "step": step,
                "total_loss": float(total_loss.item()),
                "reward_loss": float(reward_loss.item()),
                "reward_log_probability": float(reward_log_prob.mean().item()),
                "reward_target_probability": float(reward_log_prob.exp().mean().item()),
                "anchor_loss": float(anchor_loss.item()),
                "grad_norm": float(grad_norm.item()),
            }
            logs.append(record)
            print(
                f"[finetune] method={method} step={step:05d} "
                f"total={record['total_loss']:.4f} reward_logp={record['reward_log_probability']:.4f} "
                f"reward_p={record['reward_target_probability']:.4f} anchor={record['anchor_loss']:.6f} "
                f"grad={record['grad_norm']:.4f}"
            )

    return logs


@torch.no_grad()
def evaluate_generator(
    method: str,
    generator: nn.Module,
    reward_model: RewardCNN,
    eval_model: RewardCNN,
    gen_args: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float | int | str]:
    generator.eval()
    fid = build_mnist_fid(gen_args.data_dir, gen_args.num_workers, device, target_class=args.target_class)
    remaining = args.eval_num_samples
    offset = 0

    reward_correct = 0.0
    reward_target_prob_sum = 0.0
    reward_target_log_prob_sum = 0.0
    eval_correct = 0.0
    eval_target_prob_sum = 0.0
    eval_target_log_prob_sum = 0.0
    saved_soft: list[torch.Tensor] = []
    saved_hard: list[torch.Tensor] = []

    while remaining > 0:
        batch_size = min(args.eval_batch_size, remaining)
        labels = make_eval_labels(batch_size, args.target_class, device, offset=offset % NUM_CLASSES)
        z_tokens = torch.randn(batch_size, FLOW_TOKENS, gen_args.latent_channels, device=device)
        logits = generator(z_tokens, labels)
        soft_samples = torch.sigmoid(logits).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        hard_samples = (soft_samples >= 0.5).float()

        reward_metrics = classifier_alignment_metrics(reward_model, hard_samples, labels)
        reward_correct += reward_metrics["accuracy"] * batch_size
        reward_target_prob_sum += reward_metrics["target_probability"] * batch_size
        reward_target_log_prob_sum += reward_metrics["target_log_probability"] * batch_size

        eval_metrics = classifier_alignment_metrics(eval_model, hard_samples, labels)
        eval_correct += eval_metrics["accuracy"] * batch_size
        eval_target_prob_sum += eval_metrics["target_probability"] * batch_size
        eval_target_log_prob_sum += eval_metrics["target_log_probability"] * batch_size

        fid.update(_gray_to_rgb(_to_uint8_0_255(hard_samples)), real=False)

        current_saved = sum(chunk.shape[0] for chunk in saved_hard)
        to_save = max(0, min(batch_size, args.save_num_samples - current_saved))
        if to_save > 0:
            saved_soft.append(soft_samples[:to_save].cpu())
            saved_hard.append(hard_samples[:to_save].cpu())

        remaining -= batch_size
        offset += batch_size

    method_dir = output_dir / method
    ensure_dir(method_dir)
    if saved_hard:
        soft_grid = torch.cat(saved_soft, dim=0)
        hard_grid = torch.cat(saved_hard, dim=0)
        nrow = max(1, min(8, hard_grid.shape[0]))
        save_image(soft_grid, method_dir / "soft_grid.png", nrow=nrow)
        save_image(hard_grid, method_dir / "hard_grid.png", nrow=nrow)

    count = max(1, args.eval_num_samples)
    metrics = {
        "method": method,
        "fid": compute_fid_metric_value(fid),
        "reward_alignment_accuracy": reward_correct / count,
        "reward_target_probability": reward_target_prob_sum / count,
        "reward_target_log_probability": reward_target_log_prob_sum / count,
        "eval_alignment_accuracy": eval_correct / count,
        "eval_target_probability": eval_target_prob_sum / count,
        "eval_target_log_probability": eval_target_log_prob_sum / count,
        "num_samples": args.eval_num_samples,
        "generator_nfe": 1,
    }
    print(
        f"[eval] method={method} class_fid={metrics['fid']:.4f} "
        f"eval_acc={metrics['eval_alignment_accuracy']:.4f} "
        f"reward_acc={metrics['reward_alignment_accuracy']:.4f}"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reward fine-tune a conditional one-step MNIST generator.")
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--reward-checkpoint", type=str, required=True)
    parser.add_argument("--eval-classifier-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs/reward_finetune")
    parser.add_argument("--methods", nargs="+", choices=FINETUNE_METHODS, default=list(FINETUNE_METHODS))
    parser.add_argument("--target-class", type=parse_target_class, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-reward", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--relaxed-method", choices=("binary-concrete", "st"), default="binary-concrete")
    parser.add_argument("--relaxed-temperature", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-num-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--save-num-samples", type=int, default=64)
    parser.add_argument("--fid-num-workers", type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.eval_num_samples <= 0:
        raise ValueError("batch/eval sample counts must be positive")
    if args.lambda_reward < 0 or args.lambda_anchor < 0:
        raise ValueError("loss weights must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()

    generator_checkpoint = Path(args.generator_checkpoint).expanduser().resolve()
    reward_checkpoint = Path(args.reward_checkpoint).expanduser().resolve()
    eval_checkpoint = None if args.eval_classifier_checkpoint is None else Path(args.eval_classifier_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    ensure_dir(output_dir)
    ensure_dir(checkpoints_dir)

    gen_args = load_generator_training_args(generator_checkpoint)
    if not is_cond_mode(gen_args):
        raise ValueError("Reward fine-tuning requires a conditional generator checkpoint (`mode=cond`).")
    if args.fid_num_workers is not None:
        gen_args.num_workers = args.fid_num_workers
    gen_args.output_dir = str(output_dir)
    gen_args.checkpoint_path = None

    vae, flow, pretrained_generator = make_models(gen_args, device)
    load_generator_checkpoint(generator_checkpoint, vae, flow, pretrained_generator, device)
    freeze_module(vae)
    freeze_module(flow)
    freeze_module(pretrained_generator)

    reward_model, _ = load_classifier_model(reward_checkpoint, device)
    if eval_checkpoint is None:
        eval_model = reward_model
        eval_checkpoint_name = str(reward_checkpoint)
    else:
        eval_model, _ = load_classifier_model(eval_checkpoint, device)
        eval_checkpoint_name = str(eval_checkpoint)
    freeze_module(reward_model)
    if eval_model is not reward_model:
        freeze_module(eval_model)

    metrics = {
        "generator_checkpoint": str(generator_checkpoint),
        "reward_checkpoint": str(reward_checkpoint),
        "eval_classifier_checkpoint": eval_checkpoint_name,
        "target_class": args.target_class,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_reward": args.lambda_reward,
        "lambda_anchor": args.lambda_anchor,
        "relaxed_method": args.relaxed_method,
        "relaxed_temperature": args.relaxed_temperature,
        "fid_reference_class": args.target_class,
        "methods": {},
    }

    for method in args.methods:
        print(f"[finetune] starting method={method}")
        generator = copy.deepcopy(pretrained_generator).to(device)
        for param in generator.parameters():
            param.requires_grad_(True)

        start_time = time.perf_counter()
        logs = train_one_method(method, generator, pretrained_generator, reward_model, gen_args, args, device)
        checkpoint_path = checkpoints_dir / f"{method}.pt"
        save_finetuned_checkpoint(checkpoint_path, vae, flow, generator, gen_args, args, method, args.steps)
        method_metrics = evaluate_generator(method, generator, reward_model, eval_model, gen_args, args, device, output_dir)
        elapsed = time.perf_counter() - start_time
        method_metrics["checkpoint_path"] = str(checkpoint_path)
        method_metrics["train_log"] = logs
        method_metrics["seconds"] = elapsed
        method_metrics["seconds_per_1k"] = elapsed * 1000.0 / args.eval_num_samples
        method_metrics["finetune_steps"] = args.steps
        metrics["methods"][method] = method_metrics

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"[metrics] saved {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
