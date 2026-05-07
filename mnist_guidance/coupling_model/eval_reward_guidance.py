import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from one_step_mnist import (
    FLOW_TOKENS,
    IMAGE_SIZE,
    NUM_CLASSES,
    _gray_to_rgb,
    _to_uint8_0_255,
    build_mnist_fid,
    build_parser as build_generator_parser,
    compute_fid_metric_value,
    ensure_dir,
    generator_cfg_logits,
    get_device,
    is_cond_mode,
    load_checkpoint as load_generator_checkpoint,
    make_models,
    set_seed,
)
from train_reward_model import RewardCNN, load_checkpoint as load_classifier_checkpoint


GUIDANCE_METHODS = ("soft-latent", "relaxed-latent")


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


def load_generator_training_args(checkpoint_path: Path) -> argparse.Namespace:
    defaults = vars(build_generator_parser().parse_args([]))
    payload = torch.load(checkpoint_path, map_location="cpu")
    merged = dict(defaults)
    merged.update(payload["args"])
    return argparse.Namespace(**merged)


def load_classifier_model(checkpoint_path: Path, device: torch.device) -> tuple[RewardCNN, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    args = payload.get("args", {})
    model = RewardCNN(
        num_classes=NUM_CLASSES,
        width=args.get("model_width", 64),
        dropout=args.get("dropout", 0.1),
    ).to(device)
    load_classifier_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def target_log_prob(classifier_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(classifier_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)


def binary_concrete_samples(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    logistic_noise = torch.log(u) - torch.log1p(-u)
    return torch.sigmoid((logits + logistic_noise) / temperature)


def straight_through_relaxed(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    hard = (probs >= 0.5).to(probs.dtype)
    return hard + probs - probs.detach()


def reward_inputs_from_logits(logits: torch.Tensor, method: str, args: argparse.Namespace) -> torch.Tensor:
    if method == "soft-latent":
        samples = torch.sigmoid(logits)
    elif method == "relaxed-latent":
        if args.relaxed_method == "binary-concrete":
            samples = binary_concrete_samples(logits, temperature=args.relaxed_temperature)
        elif args.relaxed_method == "st":
            samples = straight_through_relaxed(logits)
        else:
            raise ValueError(f"Unknown relaxed method: {args.relaxed_method}")
    else:
        raise ValueError(f"Unknown guidance method: {method}")
    return samples.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


def optimize_latents(
    generator,
    reward_model: RewardCNN,
    initial_z: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor | list[float]]:
    z_tokens = initial_z.clone().detach()
    trace_objective: list[float] = []
    trace_reward: list[float] = []
    trace_target_prob: list[float] = []
    trace_grad_norm: list[float] = []

    for _ in range(args.guidance_steps):
        z_tokens = z_tokens.detach().requires_grad_(True)
        pixel_logits = generator_cfg_logits(generator, z_tokens, labels, cfg_scale=args.cfg_scale)
        reward_inputs = reward_inputs_from_logits(pixel_logits, method, args)
        classifier_logits = reward_model(reward_inputs)
        log_prob = target_log_prob(classifier_logits, labels)
        latent_penalty = z_tokens.square().flatten(1).sum(dim=1)
        objective = log_prob - args.latent_l2 * latent_penalty
        objective_mean = objective.mean()
        grad = torch.autograd.grad(objective_mean, z_tokens)[0]
        grad_norm = grad.flatten(1).norm(dim=1).mean()

        with torch.no_grad():
            z_tokens = z_tokens + args.guidance_lr * grad

        trace_objective.append(objective_mean.item())
        trace_reward.append(log_prob.mean().item())
        trace_target_prob.append(log_prob.exp().mean().item())
        trace_grad_norm.append(grad_norm.item())

    with torch.no_grad():
        final_logits = generator_cfg_logits(generator, z_tokens, labels, cfg_scale=args.cfg_scale)
        soft_samples = torch.sigmoid(final_logits).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        hard_samples = (soft_samples >= 0.5).float()
        final_reward_inputs = reward_inputs_from_logits(final_logits, method, args)
        final_classifier_logits = reward_model(final_reward_inputs)
        final_log_prob = target_log_prob(final_classifier_logits, labels)

    return {
        "z_tokens": z_tokens.detach(),
        "soft_samples": soft_samples,
        "hard_samples": hard_samples,
        "final_reward_log_prob": final_log_prob.detach(),
        "trace_objective": trace_objective,
        "trace_reward": trace_reward,
        "trace_target_prob": trace_target_prob,
        "trace_grad_norm": trace_grad_norm,
    }


@torch.no_grad()
def classifier_alignment_metrics(model: RewardCNN, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    return {
        "accuracy": (preds == labels).float().mean().item(),
        "target_probability": target_probs.mean().item(),
        "target_log_probability": target_probs.log().mean().item(),
    }


def init_method_state(
    data_dir: str,
    num_workers: int,
    target_class: int,
    device: torch.device,
) -> dict:
    fid = build_mnist_fid(data_dir, num_workers, device, target_class=target_class)
    return {
        "fid": fid,
        "num_samples": 0,
        "trace_weight": 0,
        "trace_sums": {
            "objective": None,
            "reward_log_prob": None,
            "target_probability": None,
            "grad_norm": None,
        },
        "reward_alignment_correct": 0.0,
        "reward_target_prob_sum": 0.0,
        "reward_target_log_prob_sum": 0.0,
        "eval_alignment_correct": 0.0,
        "eval_target_prob_sum": 0.0,
        "eval_target_log_prob_sum": 0.0,
        "opt_reward_log_prob_sum": 0.0,
    }


def accumulate_trace(state: dict, result: dict, batch_size: int) -> None:
    mapping = {
        "objective": result["trace_objective"],
        "reward_log_prob": result["trace_reward"],
        "target_probability": result["trace_target_prob"],
        "grad_norm": result["trace_grad_norm"],
    }
    for key, values in mapping.items():
        trace_tensor = torch.tensor(values, dtype=torch.float64)
        if state["trace_sums"][key] is None:
            state["trace_sums"][key] = trace_tensor * batch_size
        else:
            state["trace_sums"][key] += trace_tensor * batch_size
    state["trace_weight"] += batch_size


def finalize_trace(state: dict) -> dict[str, list[float]]:
    traces: dict[str, list[float]] = {}
    for key, tensor in state["trace_sums"].items():
        if tensor is None or state["trace_weight"] == 0:
            traces[key] = []
        else:
            traces[key] = (tensor / state["trace_weight"]).tolist()
    return traces


def save_method_grids(output_dir: Path, method: str, soft_samples: torch.Tensor, hard_samples: torch.Tensor, nrow: int) -> None:
    method_dir = output_dir / method
    ensure_dir(method_dir)
    save_image(soft_samples, method_dir / "soft_grid.png", nrow=nrow)
    save_image(hard_samples, method_dir / "hard_grid.png", nrow=nrow)


def save_comparison_grid(output_dir: Path, hard_samples_by_method: dict[str, torch.Tensor], nrow: int) -> None:
    left = hard_samples_by_method["soft-latent"]
    right = hard_samples_by_method["relaxed-latent"]
    pairwise = torch.cat([left, right], dim=3)
    save_image(pairwise, output_dir / "comparison_hard_grid.png", nrow=nrow)


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate training-free reward guidance on a frozen MNIST generator.")
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--reward-checkpoint", type=str, required=True)
    parser.add_argument("--eval-classifier-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs/reward_guidance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-class", type=parse_target_class, default=9)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--guidance-steps", type=int, default=20)
    parser.add_argument("--guidance-lr", type=float, default=0.05)
    parser.add_argument("--latent-l2", type=float, default=1e-4)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--relaxed-method", choices=("binary-concrete", "st"), default="binary-concrete")
    parser.add_argument("--relaxed-temperature", type=float, default=0.5)
    parser.add_argument("--save-num-samples", type=int, default=64)
    parser.add_argument("--fid-num-workers", type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    if args.guidance_steps <= 0:
        raise ValueError("guidance_steps must be positive")
    if args.save_num_samples < 0:
        raise ValueError("save_num_samples must be non-negative")


def main() -> None:
    args = build_eval_parser().parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()

    generator_checkpoint = Path(args.generator_checkpoint).expanduser().resolve()
    reward_checkpoint = Path(args.reward_checkpoint).expanduser().resolve()
    eval_checkpoint = None
    if args.eval_classifier_checkpoint is not None:
        eval_checkpoint = Path(args.eval_classifier_checkpoint).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    gen_args = load_generator_training_args(generator_checkpoint)
    if not is_cond_mode(gen_args):
        raise ValueError("Reward guidance requires a conditional generator checkpoint (`mode=cond`).")
    if args.fid_num_workers is not None:
        gen_args.num_workers = args.fid_num_workers

    vae, flow, generator = make_models(gen_args, device)
    load_generator_checkpoint(generator_checkpoint, vae, flow, generator, device)

    reward_model, _ = load_classifier_model(reward_checkpoint, device)
    if eval_checkpoint is None:
        eval_model = reward_model
        eval_checkpoint_name = str(reward_checkpoint)
    else:
        eval_model, _ = load_classifier_model(eval_checkpoint, device)
        eval_checkpoint_name = str(eval_checkpoint)

    freeze_module(generator)
    freeze_module(reward_model)
    if eval_model is not reward_model:
        freeze_module(eval_model)

    method_states = {
        method: init_method_state(gen_args.data_dir, gen_args.num_workers, args.target_class, device)
        for method in GUIDANCE_METHODS
    }
    saved_soft: dict[str, list[torch.Tensor]] = {method: [] for method in GUIDANCE_METHODS}
    saved_hard: dict[str, list[torch.Tensor]] = {method: [] for method in GUIDANCE_METHODS}

    start_time = time.perf_counter()
    remaining = args.num_samples
    label_offset = 0
    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        initial_z = torch.randn(batch_size, FLOW_TOKENS, gen_args.latent_channels, device=device)
        labels = make_eval_labels(batch_size, args.target_class, device, offset=label_offset)
        label_offset = (label_offset + batch_size) % NUM_CLASSES

        for method in GUIDANCE_METHODS:
            result = optimize_latents(generator, reward_model, initial_z, labels, method, args)
            hard_samples = result["hard_samples"]
            soft_samples = result["soft_samples"]

            state = method_states[method]
            state["num_samples"] += batch_size
            state["opt_reward_log_prob_sum"] += result["final_reward_log_prob"].sum().item()
            accumulate_trace(state, result, batch_size)

            reward_metrics = classifier_alignment_metrics(reward_model, hard_samples, labels)
            state["reward_alignment_correct"] += reward_metrics["accuracy"] * batch_size
            state["reward_target_prob_sum"] += reward_metrics["target_probability"] * batch_size
            state["reward_target_log_prob_sum"] += reward_metrics["target_log_probability"] * batch_size

            eval_metrics = classifier_alignment_metrics(eval_model, hard_samples, labels)
            state["eval_alignment_correct"] += eval_metrics["accuracy"] * batch_size
            state["eval_target_prob_sum"] += eval_metrics["target_probability"] * batch_size
            state["eval_target_log_prob_sum"] += eval_metrics["target_log_probability"] * batch_size

            state["fid"].update(_gray_to_rgb(_to_uint8_0_255(hard_samples)), real=False)

            current_saved = sum(chunk.shape[0] for chunk in saved_hard[method])
            to_save = max(0, min(batch_size, args.save_num_samples - current_saved))
            if to_save > 0:
                saved_soft[method].append(soft_samples[:to_save].detach().cpu())
                saved_hard[method].append(hard_samples[:to_save].detach().cpu())

        remaining -= batch_size
    elapsed = time.perf_counter() - start_time

    metrics = {
        "generator_checkpoint": str(generator_checkpoint),
        "reward_checkpoint": str(reward_checkpoint),
        "eval_classifier_checkpoint": eval_checkpoint_name,
        "target_class": args.target_class,
        "num_samples": args.num_samples,
        "guidance_steps": args.guidance_steps,
        "guidance_lr": args.guidance_lr,
        "latent_l2": args.latent_l2,
        "cfg_scale": args.cfg_scale,
        "relaxed_method": args.relaxed_method,
        "relaxed_temperature": args.relaxed_temperature,
        "fid_reference_class": args.target_class,
        "seconds": elapsed,
        "seconds_per_1k": elapsed * 1000.0 / args.num_samples,
        "methods": {},
    }

    hard_samples_for_grid: dict[str, torch.Tensor] = {}
    for method in GUIDANCE_METHODS:
        state = method_states[method]
        count = max(1, state["num_samples"])
        method_metrics = {
            "fid": compute_fid_metric_value(state["fid"]),
            "optimization_reward_log_prob": state["opt_reward_log_prob_sum"] / count,
            "reward_alignment_accuracy": state["reward_alignment_correct"] / count,
            "reward_target_probability": state["reward_target_prob_sum"] / count,
            "reward_target_log_probability": state["reward_target_log_prob_sum"] / count,
            "eval_alignment_accuracy": state["eval_alignment_correct"] / count,
            "eval_target_probability": state["eval_target_prob_sum"] / count,
            "eval_target_log_probability": state["eval_target_log_prob_sum"] / count,
            "generator_nfe": args.guidance_steps + 1,
            "reward_evals": args.guidance_steps + 2,
            "trace": finalize_trace(state),
        }
        metrics["methods"][method] = method_metrics

        if saved_hard[method]:
            soft_grid_samples = torch.cat(saved_soft[method], dim=0)
            hard_grid_samples = torch.cat(saved_hard[method], dim=0)
            hard_samples_for_grid[method] = hard_grid_samples
            nrow = max(1, min(8, hard_grid_samples.shape[0]))
            save_method_grids(output_dir, method, soft_grid_samples, hard_grid_samples, nrow=nrow)

    if len(hard_samples_for_grid) == len(GUIDANCE_METHODS):
        nrow = max(1, min(8, hard_samples_for_grid["soft-latent"].shape[0]))
        save_comparison_grid(output_dir, hard_samples_for_grid, nrow=nrow)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    for method in GUIDANCE_METHODS:
        method_metrics = metrics["methods"][method]
        print(
            f"[reward-guidance] method={method} "
            f"class_fid={method_metrics['fid']:.4f} "
            f"eval_acc={method_metrics['eval_alignment_accuracy']:.4f} "
            f"reward_acc={method_metrics['reward_alignment_accuracy']:.4f}"
        )
if __name__ == "__main__":
    main()
