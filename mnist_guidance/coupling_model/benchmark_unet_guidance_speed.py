from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval_reward_guidance import load_classifier_model, load_generator_training_args, optimize_latents
from mdm_baseline.eval_unet_mdm_cfg import sample_unet_cfg
from mdm_baseline.eval_unet_mdm_classifier_guidance import sample_unet_classifier_guidance
from mdm_baseline.train_unet_mdm import build_parser as build_unet_train_parser
from mdm_baseline.unet_model import load_unet_checkpoint, make_unet_mdm_from_args
from mdm_baseline.utils import load_mdm_training_args, make_eval_labels
from one_step_mnist import (
    FLOW_TOKENS,
    NUM_CLASSES,
    generate_binary_samples,
    get_device,
    is_cond_mode,
    load_checkpoint as load_generator_checkpoint,
    make_models,
    set_seed,
)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seconds_per_1k(start: float, num_samples: int) -> float:
    return (time.perf_counter() - start) * 1000.0 / num_samples


def benchmark_one_step_cfg(generator, gen_args, args, device: torch.device) -> list[dict]:
    rows = []
    for scale in args.cfg_scales:
        remaining = args.num_samples
        offset = 0
        cuda_sync(device)
        start = time.perf_counter()
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            labels = make_eval_labels(batch_size, None, device, offset=offset)
            offset = (offset + batch_size) % NUM_CLASSES
            z_tokens = torch.randn(batch_size, FLOW_TOKENS, gen_args.latent_channels, device=device)
            _ = generate_binary_samples(generator, z_tokens, labels, cfg_scale=scale)
            remaining -= batch_size
        cuda_sync(device)
        rows.append(
            {
                "model": "One-step",
                "guidance": "CFG",
                "setting": f"s={scale:g}",
                "seconds_per_1k": seconds_per_1k(start, args.num_samples),
            }
        )
    return rows


def benchmark_one_step_reward(generator, reward_model, gen_args, args, device: torch.device) -> list[dict]:
    rows = []
    method_names = {"soft-latent": "soft", "relaxed-latent": "gumbel"}
    for method in ["soft-latent", "relaxed-latent"]:
        remaining = args.num_samples
        offset = 0
        cuda_sync(device)
        start = time.perf_counter()
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            labels = make_eval_labels(batch_size, None, device, offset=offset)
            offset = (offset + batch_size) % NUM_CLASSES
            z_tokens = torch.randn(batch_size, FLOW_TOKENS, gen_args.latent_channels, device=device)
            _ = optimize_latents(generator, reward_model, z_tokens, labels, method, args)
            remaining -= batch_size
        cuda_sync(device)
        rows.append(
            {
                "model": "One-step",
                "guidance": "Classifier guidance",
                "setting": f"{method_names[method]}, K={args.guidance_steps}",
                "seconds_per_1k": seconds_per_1k(start, args.num_samples),
            }
        )
    return rows


def benchmark_one_step_ft(checkpoint_paths: list[Path], args, device: torch.device) -> list[dict]:
    rows = []
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            continue
        gen_args = load_generator_training_args(checkpoint_path)
        vae, flow, generator = make_models(gen_args, device)
        load_generator_checkpoint(checkpoint_path, vae, flow, generator, device)
        generator.eval()
        method = checkpoint_path.stem
        method_label = {"ft-soft": "soft", "ft-relaxed": "gumbel"}.get(method, method)
        remaining = args.num_samples
        offset = 0
        cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            while remaining > 0:
                batch_size = min(args.batch_size, remaining)
                labels = make_eval_labels(batch_size, None, device, offset=offset)
                offset = (offset + batch_size) % NUM_CLASSES
                z_tokens = torch.randn(batch_size, FLOW_TOKENS, gen_args.latent_channels, device=device)
                _ = generate_binary_samples(generator, z_tokens, labels, cfg_scale=1.0)
                remaining -= batch_size
        cuda_sync(device)
        rows.append(
            {
                "model": "One-step",
                "guidance": "Reward fine-tuning",
                "setting": f"{method_label}, T={args.ft_steps}",
                "seconds_per_1k": seconds_per_1k(start, args.num_samples),
            }
        )
    return rows


def benchmark_unet_cfg(model, args, device: torch.device) -> list[dict]:
    rows = []
    for scale in args.cfg_scales:
        remaining = args.num_samples
        offset = 0
        cuda_sync(device)
        start = time.perf_counter()
        while remaining > 0:
            batch_size = min(args.unet_batch_size, remaining)
            labels = make_eval_labels(batch_size, None, device, offset=offset)
            offset = (offset + batch_size) % NUM_CLASSES
            _ = sample_unet_cfg(
                model,
                labels,
                steps=args.unet_steps,
                cfg_scale=scale,
                temperature=1.0,
                argmax=False,
                sampler=args.sampler,
            )
            remaining -= batch_size
        cuda_sync(device)
        rows.append(
            {
                "model": "MDM",
                "guidance": "CFG",
                "setting": f"s={scale:g}",
                "seconds_per_1k": seconds_per_1k(start, args.num_samples),
            }
        )
    return rows


def benchmark_unet_classifier(model, reward_model, args, device: torch.device) -> list[dict]:
    rows = []
    for scale in args.classifier_scales:
        remaining = args.num_samples
        offset = 0
        cuda_sync(device)
        start = time.perf_counter()
        while remaining > 0:
            batch_size = min(args.unet_batch_size, remaining)
            labels = make_eval_labels(batch_size, None, device, offset=offset)
            offset = (offset + batch_size) % NUM_CLASSES
            _ = sample_unet_classifier_guidance(
                model,
                reward_model,
                labels,
                steps=args.unet_steps,
                guidance_scale=scale,
                temperature=1.0,
                argmax=False,
                sampler=args.sampler,
            )
            remaining -= batch_size
        cuda_sync(device)
        rows.append(
            {
                "model": "MDM",
                "guidance": "Classifier guidance",
                "setting": f"s={scale:g}",
                "seconds_per_1k": seconds_per_1k(start, args.num_samples),
            }
        )
    return rows


def benchmark_unet_ft(checkpoint_path: Path, train_args, args, device: torch.device) -> list[dict]:
    if not checkpoint_path.exists():
        return []
    model = make_unet_mdm_from_args(train_args, device)
    load_unet_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    remaining = args.num_samples
    offset = 0
    cuda_sync(device)
    start = time.perf_counter()
    while remaining > 0:
        batch_size = min(args.unet_batch_size, remaining)
        labels = make_eval_labels(batch_size, None, device, offset=offset)
        offset = (offset + batch_size) % NUM_CLASSES
        _ = sample_unet_cfg(
            model,
            labels,
            steps=args.unet_steps,
            cfg_scale=1.0,
            temperature=1.0,
            argmax=False,
            sampler=args.sampler,
        )
        remaining -= batch_size
    cuda_sync(device)
    return [
        {
            "model": "MDM",
            "guidance": "Reward fine-tuning",
            "setting": f"DRAKES, T={args.ft_steps}",
            "seconds_per_1k": seconds_per_1k(start, args.num_samples),
        }
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark pure sampling speed for one-step and U-Net MDM guidance methods.")
    parser.add_argument("--one-step-checkpoint", type=str, required=True)
    parser.add_argument("--unet-checkpoint", type=str, required=True)
    parser.add_argument("--reward-checkpoint", type=str, required=True)
    parser.add_argument("--one-step-ft-dir", type=str, default=None)
    parser.add_argument("--unet-drakes-checkpoint", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unet-batch-size", type=int, default=64)
    parser.add_argument("--unet-steps", type=int, default=256)
    parser.add_argument("--sampler", choices=("md4", "confidence"), default="md4")
    parser.add_argument("--guidance-steps", type=int, default=5)
    parser.add_argument("--guidance-lr", type=float, default=0.05)
    parser.add_argument("--latent-l2", type=float, default=1e-4)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--relaxed-method", choices=("binary-concrete", "st"), default="binary-concrete")
    parser.add_argument("--relaxed-temperature", type=float, default=0.5)
    parser.add_argument("--cfg-scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--classifier-scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--ft-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device()
    rows: list[dict] = []

    one_step_checkpoint = Path(args.one_step_checkpoint).expanduser().resolve()
    gen_args = load_generator_training_args(one_step_checkpoint)
    if not is_cond_mode(gen_args):
        raise ValueError("Expected a conditional one-step checkpoint.")
    vae, flow, generator = make_models(gen_args, device)
    load_generator_checkpoint(one_step_checkpoint, vae, flow, generator, device)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad_(False)

    reward_model, _ = load_classifier_model(Path(args.reward_checkpoint).expanduser().resolve(), device)
    rows.extend(benchmark_one_step_cfg(generator, gen_args, args, device))
    rows.extend(benchmark_one_step_reward(generator, reward_model, gen_args, args, device))

    if args.one_step_ft_dir is not None:
        ft_dir = Path(args.one_step_ft_dir).expanduser().resolve()
        rows.extend(benchmark_one_step_ft([ft_dir / "ft-soft.pt", ft_dir / "ft-relaxed.pt"], args, device))

    unet_checkpoint = Path(args.unet_checkpoint).expanduser().resolve()
    train_args = load_mdm_training_args(unet_checkpoint, build_unet_train_parser())
    unet = make_unet_mdm_from_args(train_args, device)
    load_unet_checkpoint(unet_checkpoint, unet, optimizer=None, device=device)
    unet.eval()
    for param in unet.parameters():
        param.requires_grad_(False)
    rows.extend(benchmark_unet_cfg(unet, args, device))
    rows.extend(benchmark_unet_classifier(unet, reward_model, args, device))

    if args.unet_drakes_checkpoint is not None:
        rows.extend(benchmark_unet_ft(Path(args.unet_drakes_checkpoint).expanduser().resolve(), train_args, args, device))

    payload = {"num_samples": args.num_samples, "seed": args.seed, "rows": rows}
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {output_path}")
    for row in rows:
        print(f"[speed] {row['model']} {row['guidance']} {row['setting']} {row['seconds_per_1k']:.4f} sec/1k")


if __name__ == "__main__":
    main()
