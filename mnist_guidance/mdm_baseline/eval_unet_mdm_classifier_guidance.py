from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one_step_mnist import NUM_CLASSES, ensure_dir, get_device, set_seed  # noqa: E402
from mdm_baseline.eval_unet_mdm_cfg import bernoulli_log_probs, parse_target_class  # noqa: E402
from mdm_baseline.model import MASK_TOKEN  # noqa: E402
from mdm_baseline.train_unet_mdm import build_parser as build_train_parser, md4_masking_schedule  # noqa: E402
from mdm_baseline.unet_model import UNET_IMAGE_SIZE, load_unet_checkpoint, make_unet_mdm_from_args  # noqa: E402
from mdm_baseline.utils import (  # noqa: E402
    classifier_alignment_metrics,
    fid_value,
    load_classifier_model,
    load_mdm_training_args,
    make_eval_labels,
    new_fid,
    save_sample_grid,
    update_fid_with_images,
    write_json,
)


def compose_soft_image(tokens: torch.Tensor, probs_one: torch.Tensor) -> torch.Tensor:
    masked = tokens == MASK_TOKEN
    soft_flat = torch.where(masked, probs_one, tokens.float().clamp(0, 1))
    return soft_flat.view(-1, 1, UNET_IMAGE_SIZE, UNET_IMAGE_SIZE)


def classifier_guided_log_probs(
    model,
    classifier: torch.nn.Module,
    tokens: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(tokens, t, labels)
    probs_one = torch.sigmoid(logits)
    x_soft = compose_soft_image(tokens, probs_one).detach().requires_grad_(True)
    reward = F.log_softmax(classifier(x_soft), dim=-1).gather(1, labels[:, None]).squeeze(1).mean()
    grad = torch.autograd.grad(reward, x_soft)[0].flatten(1)
    guided_logits = logits + guidance_scale * grad * (tokens == MASK_TOKEN)
    return bernoulli_log_probs(guided_logits, temperature), reward.detach()


def sample_unet_classifier_guidance(
    model,
    classifier: torch.nn.Module,
    labels: torch.Tensor,
    steps: int,
    guidance_scale: float,
    temperature: float,
    argmax: bool,
    sampler: str,
) -> tuple[torch.Tensor, list[float]]:
    tokens = torch.full((labels.shape[0], model.num_pixels), MASK_TOKEN, device=labels.device, dtype=torch.long)
    trace: list[float] = []
    if sampler == "md4":
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
            log_probs, reward = classifier_guided_log_probs(
                model,
                classifier,
                tokens,
                ti,
                labels,
                guidance_scale=guidance_scale,
                temperature=temperature,
            )
            trace.append(float(reward.item()))
            probs = log_probs.softmax(dim=-1)
            samples = log_probs.argmax(dim=-1) if argmax else torch.distributions.Categorical(probs=probs).sample()
            reveal = (torch.rand_like(tokens.float()) < unmask_prob[:, None]) & masked
            tokens = torch.where(reveal, samples, tokens)
    elif sampler == "confidence":
        for step in range(steps):
            masked = tokens == MASK_TOKEN
            remaining = masked.sum(dim=1)
            if int(remaining.max().item()) == 0:
                break
            t = (remaining.float() / model.num_pixels).clamp(1e-4, 1 - 1e-4)
            log_probs, reward = classifier_guided_log_probs(
                model,
                classifier,
                tokens,
                t,
                labels,
                guidance_scale=guidance_scale,
                temperature=temperature,
            )
            trace.append(float(reward.item()))
            probs = log_probs.softmax(dim=-1)
            samples = log_probs.argmax(dim=-1) if argmax else torch.distributions.Categorical(probs=probs).sample()
            confidence = probs.max(dim=-1).values.masked_fill(~masked, -1.0)
            reveal_count = int(torch.ceil(remaining.float() / (steps - step)).max().item())
            _, indices = confidence.topk(k=min(reveal_count, int(remaining.max().item())), dim=1)
            reveal = torch.zeros_like(masked)
            reveal.scatter_(1, indices, True)
            reveal &= masked
            tokens = torch.where(reveal, samples, tokens)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    tokens = tokens.masked_fill(tokens == MASK_TOKEN, 0)
    return tokens.float().view(-1, 1, UNET_IMAGE_SIZE, UNET_IMAGE_SIZE), trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 32x32 U-Net MDM with classifier guidance.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--reward-checkpoint", type=str, required=True)
    parser.add_argument("--eval-classifier-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs/unet_mdm_classifier_guidance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-class", type=parse_target_class, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--sampler", choices=("md4", "confidence"), default="md4")
    parser.add_argument("--guidance-scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--save-num-samples", type=int, default=100)
    parser.add_argument("--fid-num-workers", type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 1 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("--num-samples must be > 1 and --batch-size/--steps must be positive")
    if args.save_num_samples < 0:
        raise ValueError("--save-num-samples must be non-negative")


def evaluate_scale(model, reward_model, eval_model, train_args: argparse.Namespace, args: argparse.Namespace, device: torch.device, guidance_scale: float) -> dict:
    start_time = time.perf_counter()
    num_workers = train_args.num_workers if args.fid_num_workers is None else args.fid_num_workers
    fid = new_fid(train_args.data_dir, num_workers, device, target_class=args.target_class)
    remaining = args.num_samples
    label_offset = 0
    saved: list[torch.Tensor] = []
    reward_trace_sum: torch.Tensor | None = None
    reward_trace_weight = 0
    reward_correct = 0.0
    reward_target_prob = 0.0
    reward_target_log_prob = 0.0
    eval_correct = 0.0
    eval_target_prob = 0.0
    eval_target_log_prob = 0.0

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        labels = make_eval_labels(batch_size, args.target_class, device, offset=label_offset)
        label_offset = (label_offset + batch_size) % NUM_CLASSES
        images, trace = sample_unet_classifier_guidance(
            model,
            reward_model,
            labels,
            steps=args.steps,
            guidance_scale=guidance_scale,
            temperature=args.temperature,
            argmax=args.argmax,
            sampler=args.sampler,
        )
        update_fid_with_images(fid, images)
        reward_metrics = classifier_alignment_metrics(reward_model, images, labels)
        reward_correct += reward_metrics["accuracy"] * batch_size
        reward_target_prob += reward_metrics["target_probability"] * batch_size
        reward_target_log_prob += reward_metrics["target_log_probability"] * batch_size
        eval_metrics = classifier_alignment_metrics(eval_model, images, labels)
        eval_correct += eval_metrics["accuracy"] * batch_size
        eval_target_prob += eval_metrics["target_probability"] * batch_size
        eval_target_log_prob += eval_metrics["target_log_probability"] * batch_size
        trace_tensor = torch.tensor(trace, dtype=torch.float64)
        if reward_trace_sum is None:
            reward_trace_sum = trace_tensor * batch_size
        else:
            reward_trace_sum += trace_tensor * batch_size
        reward_trace_weight += batch_size
        current_saved = sum(chunk.shape[0] for chunk in saved)
        to_save = max(0, min(batch_size, args.save_num_samples - current_saved))
        if to_save > 0:
            saved.append(images[:to_save].detach().cpu())
        remaining -= batch_size

    scale_dir = Path(args.output_dir) / f"classifier_guidance_{guidance_scale:g}"
    if saved:
        save_sample_grid(scale_dir, "samples.png", torch.cat(saved, dim=0), nrow=10)
    count = max(1, args.num_samples)
    trace_values = [] if reward_trace_sum is None else (reward_trace_sum / max(1, reward_trace_weight)).tolist()
    elapsed = time.perf_counter() - start_time
    return {
        "fid": fid_value(fid),
        "num_samples": args.num_samples,
        "steps": args.steps,
        "sampler": args.sampler,
        "denoiser_nfe": args.steps,
        "classifier_evals": args.steps,
        "guidance_scale": guidance_scale,
        "seconds": elapsed,
        "seconds_per_1k": elapsed * 1000.0 / args.num_samples,
        "reward_alignment_accuracy": reward_correct / count,
        "reward_target_probability": reward_target_prob / count,
        "reward_target_log_probability": reward_target_log_prob / count,
        "eval_alignment_accuracy": eval_correct / count,
        "eval_target_probability": eval_target_prob / count,
        "eval_target_log_probability": eval_target_log_prob / count,
        "reward_trace": trace_values,
    }


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
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    reward_model, _ = load_classifier_model(reward_checkpoint, device)
    if args.eval_classifier_checkpoint is None:
        eval_model = reward_model
        eval_checkpoint_name = str(reward_checkpoint)
    else:
        eval_checkpoint = Path(args.eval_classifier_checkpoint).expanduser().resolve()
        eval_model, _ = load_classifier_model(eval_checkpoint, device)
        eval_checkpoint_name = str(eval_checkpoint)

    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "reward_checkpoint": str(reward_checkpoint),
        "eval_classifier_checkpoint": eval_checkpoint_name,
        "target_class": args.target_class,
        "temperature": args.temperature,
        "argmax": args.argmax,
        "sampler": args.sampler,
        "methods": {},
    }
    for scale in args.guidance_scales:
        result = evaluate_scale(model, reward_model, eval_model, train_args, args, device, guidance_scale=scale)
        metrics["methods"][f"classifier_guidance_{scale:g}"] = result
        print(
            f"[unet-mdm-classifier-guidance] scale={scale:g} fid={result['fid']:.4f} "
            f"eval_acc={result['eval_alignment_accuracy']:.4f} reward_acc={result['reward_alignment_accuracy']:.4f}"
        )

    write_json(output_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
