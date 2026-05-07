from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one_step_mnist import NUM_CLASSES, ensure_dir, get_device, set_seed  # noqa: E402
from mdm_baseline.eval_unet_mdm_cfg import parse_target_class, sample_unet_cfg  # noqa: E402
from mdm_baseline.model import MASK_TOKEN  # noqa: E402
from mdm_baseline.train_unet_mdm import build_parser as build_train_parser, md4_masking_schedule  # noqa: E402
from mdm_baseline.unet_model import UNET_IMAGE_SIZE, load_unet_checkpoint, make_unet_mdm_from_args, save_unet_checkpoint  # noqa: E402
from mdm_baseline.utils import (  # noqa: E402
    classifier_alignment_metrics,
    fid_value,
    load_classifier_model,
    load_mdm_training_args,
    make_eval_labels,
    new_fid,
    save_sample_grid,
    target_log_prob,
    update_fid_with_images,
    write_json,
)


def binary_concrete_st(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    logistic_noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((logits + logistic_noise) / temperature)
    hard = (soft >= 0.5).to(soft.dtype)
    return hard + soft - soft.detach()


def bernoulli_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_logp = torch.stack([F.logsigmoid(-student_logits), F.logsigmoid(student_logits)], dim=-1)
    teacher_prob_one = torch.sigmoid(teacher_logits)
    teacher_probs = torch.stack([1 - teacher_prob_one, teacher_prob_one], dim=-1)
    return F.kl_div(student_logp, teacher_probs, reduction="none").sum(dim=-1)


def relaxed_reverse_trajectory(
    model: nn.Module,
    teacher: nn.Module,
    labels: torch.Tensor,
    steps: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = labels.shape[0]
    tokens = torch.full((batch_size, model.num_pixels), MASK_TOKEN, device=labels.device, dtype=torch.long)
    soft_values = torch.full((batch_size, model.num_pixels), 0.5, device=labels.device)
    kl_terms: list[torch.Tensor] = []
    t_grid = torch.linspace(0, 1, steps + 1, device=labels.device)

    for i in range(steps, 1, -1):
        masked = tokens == MASK_TOKEN
        if int(masked.sum().item()) == 0:
            break
        ti = torch.full((batch_size,), t_grid[i].item(), device=labels.device)
        si = torch.full((batch_size,), t_grid[i - 1].item(), device=labels.device)
        alpha_t = md4_masking_schedule(ti)
        alpha_s = md4_masking_schedule(si)
        unmask_prob = ((alpha_s - alpha_t) / (1 - alpha_t).clamp_min(1e-6)).clamp(0, 1)

        logits = model(tokens, ti, labels)
        with torch.no_grad():
            teacher_logits = teacher(tokens, ti, labels)
        step_kl = bernoulli_kl(logits, teacher_logits)
        kl_terms.append(step_kl[masked].mean())

        relaxed = binary_concrete_st(logits, temperature=temperature)
        hard_samples = (relaxed >= 0.5).long()
        reveal = (torch.rand_like(tokens.float()) < unmask_prob[:, None]) & masked
        soft_values = torch.where(reveal, relaxed, soft_values)
        tokens = torch.where(reveal, hard_samples, tokens).detach()

    images = soft_values.view(-1, 1, UNET_IMAGE_SIZE, UNET_IMAGE_SIZE)
    kl_anchor = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=labels.device)
    return images, kl_anchor


@torch.no_grad()
def evaluate_model(model, eval_model, train_args: argparse.Namespace, args: argparse.Namespace, device: torch.device, output_dir: Path) -> dict:
    start_time = time.perf_counter()
    num_workers = train_args.num_workers if args.fid_num_workers is None else args.fid_num_workers
    fid = new_fid(train_args.data_dir, num_workers, device, target_class=args.target_class)
    remaining = args.eval_num_samples
    label_offset = 0
    saved: list[torch.Tensor] = []
    eval_correct = 0.0
    eval_target_prob = 0.0
    eval_target_log_prob = 0.0

    while remaining > 0:
        batch_size = min(args.eval_batch_size, remaining)
        labels = make_eval_labels(batch_size, args.target_class, device, offset=label_offset)
        label_offset = (label_offset + batch_size) % NUM_CLASSES
        eval_steps = args.eval_sample_steps if args.eval_sample_steps is not None else args.sample_steps
        images = sample_unet_cfg(
            model,
            labels,
            steps=eval_steps,
            cfg_scale=args.eval_cfg_scale,
            temperature=args.eval_temperature,
            argmax=args.eval_argmax,
            sampler=args.eval_sampler,
        )
        update_fid_with_images(fid, images)
        metrics = classifier_alignment_metrics(eval_model, images, labels)
        eval_correct += metrics["accuracy"] * batch_size
        eval_target_prob += metrics["target_probability"] * batch_size
        eval_target_log_prob += metrics["target_log_probability"] * batch_size

        current_saved = sum(chunk.shape[0] for chunk in saved)
        to_save = max(0, min(batch_size, args.save_num_samples - current_saved))
        if to_save > 0:
            saved.append(images[:to_save].cpu())
        remaining -= batch_size

    if saved:
        save_sample_grid(output_dir, "samples.png", torch.cat(saved, dim=0), nrow=10)
    count = max(1, args.eval_num_samples)
    elapsed = time.perf_counter() - start_time
    return {
        "fid": fid_value(fid),
        "eval_alignment_accuracy": eval_correct / count,
        "eval_target_probability": eval_target_prob / count,
        "eval_target_log_probability": eval_target_log_prob / count,
        "num_samples": args.eval_num_samples,
        "steps": eval_steps,
        "sampler": args.eval_sampler,
        "denoiser_nfe": eval_steps * (2 if args.eval_cfg_scale not in (0.0, 1.0) else 1),
        "seconds": elapsed,
        "seconds_per_1k": elapsed * 1000.0 / args.eval_num_samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRAKES-style reward fine-tuning for 32x32 U-Net MDM.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--reward-checkpoint", type=str, required=True)
    parser.add_argument("--eval-classifier-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs/unet_mdm_drakes_finetune")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-class", type=parse_target_class, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-reward", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)
    parser.add_argument("--relaxed-temperature", type=float, default=0.7)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-num-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-sample-steps", type=int, default=None)
    parser.add_argument("--eval-cfg-scale", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-argmax", action="store_true")
    parser.add_argument("--eval-sampler", choices=("md4", "confidence"), default="md4")
    parser.add_argument("--save-num-samples", type=int, default=100)
    parser.add_argument("--fid-num-workers", type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.batch_size <= 0 or args.sample_steps <= 0:
        raise ValueError("--steps, --batch-size, and --sample-steps must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    reward_checkpoint = Path(args.reward_checkpoint).expanduser().resolve()
    train_args = load_mdm_training_args(checkpoint_path, build_train_parser())
    model = make_unet_mdm_from_args(train_args, device)
    load_unet_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    reward_model, _ = load_classifier_model(reward_checkpoint, device)
    if args.eval_classifier_checkpoint is None:
        eval_model = reward_model
        eval_checkpoint_name = str(reward_checkpoint)
    else:
        eval_checkpoint = Path(args.eval_classifier_checkpoint).expanduser().resolve()
        eval_model, _ = load_classifier_model(eval_checkpoint, device)
        eval_checkpoint_name = str(eval_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    logs: list[dict] = []
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        model.train()
        labels = make_eval_labels(args.batch_size, args.target_class, device, offset=((step - 1) * args.batch_size) % NUM_CLASSES)
        images, kl_anchor = relaxed_reverse_trajectory(
            model,
            teacher,
            labels,
            steps=args.sample_steps,
            temperature=args.relaxed_temperature,
        )
        reward_logits = reward_model(images)
        reward_log_prob = target_log_prob(reward_logits, labels)
        reward_loss = -reward_log_prob.mean()
        total_loss = args.lambda_reward * reward_loss + args.lambda_anchor * kl_anchor

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "total_loss": float(total_loss.item()),
                "reward_loss": float(reward_loss.item()),
                "reward_log_probability": float(reward_log_prob.mean().item()),
                "reward_target_probability": float(reward_log_prob.exp().mean().item()),
                "kl_anchor": float(kl_anchor.item()),
                "grad_norm": float(grad_norm.item()),
            }
            logs.append(record)
            print(
                f"[unet-drakes-ft] step={step:05d} total={record['total_loss']:.4f} "
                f"reward_logp={record['reward_log_probability']:.4f} "
                f"reward_p={record['reward_target_probability']:.4f} "
                f"kl={record['kl_anchor']:.6f} grad={record['grad_norm']:.4f}"
            )

    ckpt_path = output_dir / "checkpoints" / "last.pt"
    save_unet_checkpoint(
        ckpt_path,
        model,
        optimizer,
        train_args,
        epoch=0,
        global_step=args.steps,
        extra={"finetune_args": vars(args), "finetune_logs": logs},
    )
    eval_metrics = evaluate_model(model, eval_model, train_args, args, device, output_dir)
    total_elapsed = time.perf_counter() - start_time
    metrics = {
        "pretrained_checkpoint": str(checkpoint_path),
        "finetuned_checkpoint": str(ckpt_path),
        "reward_checkpoint": str(reward_checkpoint),
        "eval_classifier_checkpoint": eval_checkpoint_name,
        "target_class": args.target_class,
        "train_logs": logs,
        "finetune_seconds": total_elapsed,
        "finetune_steps": args.steps,
        "eval": eval_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"[unet-drakes-ft-eval] fid={eval_metrics['fid']:.4f} eval_acc={eval_metrics['eval_alignment_accuracy']:.4f}")


if __name__ == "__main__":
    main()
