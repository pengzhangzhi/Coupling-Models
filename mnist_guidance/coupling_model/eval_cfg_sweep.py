import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from one_step_mnist import (
    FLOW_TOKENS,
    IMAGE_SIZE,
    NUM_CLASSES,
    build_parser,
    compute_generator_fid,
    ensure_dir,
    generate_binary_samples,
    get_device,
    is_cond_mode,
    load_checkpoint,
    make_models,
    set_seed,
)
from eval_reward_guidance import classifier_alignment_metrics, load_classifier_model


DEFAULT_FID_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
DEFAULT_VISUAL_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0]


def parse_scale_list(values: list[str]) -> list[float]:
    scales: list[float] = []
    for value in values:
        for piece in value.split(","):
            stripped = piece.strip()
            if stripped:
                scales.append(float(stripped))
    if not scales:
        raise ValueError("At least one scale must be provided.")
    return scales


def checkpoint_run_dir(output_root: Path, checkpoint_path: Path) -> Path:
    checkpoint_name = checkpoint_path.stem
    return output_root / checkpoint_name


def load_checkpoint_training_args(checkpoint_path: Path) -> argparse.Namespace:
    defaults = vars(build_parser().parse_args([]))
    payload = torch.load(checkpoint_path, map_location="cpu")
    merged = dict(defaults)
    merged.update(payload["args"])
    return argparse.Namespace(**merged)


def scale_dir_name(scale: float) -> str:
    return f"scale_{scale:.2f}"


def visual_grid_tensor(samples: torch.Tensor, samples_per_class: int) -> torch.Tensor:
    return make_grid(samples.detach().cpu(), nrow=samples_per_class, padding=2, pad_value=1.0)


def row_centers(num_rows: int, cell_size: int, padding: int = 2) -> list[float]:
    return [padding + (cell_size / 2.0) + row * (cell_size + padding) for row in range(num_rows)]


def save_fid_results(run_dir: Path, fid_results: list[dict]) -> None:
    csv_path = run_dir / "fid_results.csv"
    json_path = run_dir / "fid_results.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in fid_results for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fid_results)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(fid_results, handle, indent=2, sort_keys=True)


def plot_fid_figure(figures_dir: Path, fid_results: list[dict]) -> None:
    scales = [item["scale"] for item in fid_results]
    fids = [item["fid"] for item in fid_results]
    best_idx = min(range(len(fid_results)), key=lambda idx: fids[idx])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(scales, fids, marker="o", linewidth=1.8, markersize=5, color="black")
    ax.set_xlabel("Guidance Scale")
    ax.set_ylabel("FID")
    ax.set_title("MNIST Generator CFG Sweep")
    ax.grid(True, alpha=0.25, linewidth=0.6)

    annotations = {
        0.0: "s=0",
        1.0: "s=1",
        scales[best_idx]: f"best={scales[best_idx]:.2f}",
    }
    for scale, label in annotations.items():
        if scale not in scales:
            continue
        idx = scales.index(scale)
        ax.annotate(label, (scales[idx], fids[idx]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(figures_dir / "fid_vs_scale.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fid_vs_scale.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_visual_appendix_figure(
    figures_dir: Path,
    scale_to_grid: dict[float, torch.Tensor],
    visual_scales: list[float],
    samples_per_class: int,
) -> None:
    num_scales = len(visual_scales)
    fig, axes = plt.subplots(
        1,
        num_scales,
        figsize=(num_scales * 3.2, 11.5),
        constrained_layout=True,
    )
    if num_scales == 1:
        axes = [axes]

    y_ticks = row_centers(NUM_CLASSES, IMAGE_SIZE)
    for index, scale in enumerate(visual_scales):
        ax = axes[index]
        grid_tensor = scale_to_grid[scale]
        if grid_tensor.shape[0] == 1:
            grid = grid_tensor.squeeze(0).numpy()
        else:
            grid = grid_tensor.permute(1, 2, 0).numpy()
        ax.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"s = {scale:.2f}", fontsize=12)
        ax.set_xticks([])
        if index == 0:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=10)
            ax.set_ylabel("Class", fontsize=11)
        else:
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"{samples_per_class} samples/class", fontsize=10)

    fig.savefig(figures_dir / "visual_cfg_sweep_appendix.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "visual_cfg_sweep_appendix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep generator-side classifier-free guidance for a trained MNIST model.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--eval-classifier-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="cfg_sweep")
    parser.add_argument("--fid-scales", nargs="+", default=[str(scale) for scale in DEFAULT_FID_SCALES])
    parser.add_argument("--visual-scales", nargs="+", default=[str(scale) for scale in DEFAULT_VISUAL_SCALES])
    parser.add_argument("--fid-num-gen", type=int, default=1000)
    parser.add_argument("--visual-samples-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def main() -> None:
    cli_args = build_eval_parser().parse_args()
    checkpoint_path = Path(cli_args.checkpoint_path).expanduser().resolve()
    output_root = Path(cli_args.output_dir).expanduser().resolve()
    run_dir = checkpoint_run_dir(output_root, checkpoint_path)
    figures_dir = run_dir / "figures"
    scales_dir = run_dir / "scales"
    ensure_dir(figures_dir)
    ensure_dir(scales_dir)

    fid_scales = parse_scale_list(cli_args.fid_scales)
    visual_scales = parse_scale_list(cli_args.visual_scales)

    train_args = load_checkpoint_training_args(checkpoint_path)
    train_args.checkpoint_path = str(checkpoint_path)
    train_args.fid_num_gen = cli_args.fid_num_gen
    train_args.output_dir = str(run_dir)
    if cli_args.num_workers is not None:
        train_args.num_workers = cli_args.num_workers

    if not is_cond_mode(train_args):
        raise ValueError("CFG sweep requires a conditional checkpoint (`mode=cond`).")

    device = get_device()
    set_seed(cli_args.seed)
    vae, flow, generator = make_models(train_args, device)
    payload = load_checkpoint(checkpoint_path, vae, flow, generator, device)
    eval_model = None
    eval_checkpoint_name = None
    if cli_args.eval_classifier_checkpoint is not None:
        eval_checkpoint = Path(cli_args.eval_classifier_checkpoint).expanduser().resolve()
        eval_model, _ = load_classifier_model(eval_checkpoint, device)
        eval_checkpoint_name = str(eval_checkpoint)

    fid_results: list[dict] = []
    for scale in fid_scales:
        start_time = time.perf_counter()
        fid_value = compute_generator_fid(generator, train_args, device, cfg_scale=scale, n_gen=cli_args.fid_num_gen)
        elapsed = time.perf_counter() - start_time
        result = {
            "scale": scale,
            "fid": fid_value,
            "num_samples": cli_args.fid_num_gen,
            "seconds": elapsed,
            "seconds_per_1k": elapsed * 1000.0 / cli_args.fid_num_gen,
            "generator_nfe": 1 if scale in (0.0, 1.0) else 2,
        }
        if eval_model is not None:
            remaining = cli_args.fid_num_gen
            label_offset = 0
            eval_correct = 0.0
            eval_target_prob = 0.0
            eval_target_log_prob = 0.0
            while remaining > 0:
                batch_size = min(256, remaining)
                labels = torch.arange(NUM_CLASSES, device=device).repeat(
                    (batch_size + label_offset + NUM_CLASSES - 1) // NUM_CLASSES
                )[label_offset : label_offset + batch_size]
                label_offset = (label_offset + batch_size) % NUM_CLASSES
                z_tokens = torch.randn(batch_size, FLOW_TOKENS, train_args.latent_channels, device=device)
                samples = generate_binary_samples(generator, z_tokens, labels, cfg_scale=scale)
                metrics = classifier_alignment_metrics(eval_model, samples, labels)
                eval_correct += metrics["accuracy"] * batch_size
                eval_target_prob += metrics["target_probability"] * batch_size
                eval_target_log_prob += metrics["target_log_probability"] * batch_size
                remaining -= batch_size
            count = max(1, cli_args.fid_num_gen)
            result.update(
                {
                    "eval_alignment_accuracy": eval_correct / count,
                    "eval_target_probability": eval_target_prob / count,
                    "eval_target_log_probability": eval_target_log_prob / count,
                }
            )
        fid_results.append(result)
        acc = result.get("eval_alignment_accuracy")
        acc_text = "" if acc is None else f" eval_acc={acc:.4f}"
        print(f"[cfg-sweep] scale={scale:.2f} fid={fid_value:.4f}{acc_text} seconds={elapsed:.2f}")

    save_fid_results(run_dir, fid_results)
    plot_fid_figure(figures_dir, fid_results)

    samples_per_class = cli_args.visual_samples_per_class
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(samples_per_class)
    z_bank = torch.randn(NUM_CLASSES, samples_per_class, FLOW_TOKENS, train_args.latent_channels, device=device)

    scale_to_grid: dict[float, torch.Tensor] = {}
    generator_was_training = generator.training
    generator.eval()
    for scale in visual_scales:
        z_tokens = z_bank.view(NUM_CLASSES * samples_per_class, FLOW_TOKENS, train_args.latent_channels)
        samples = generate_binary_samples(generator, z_tokens, labels, cfg_scale=scale)
        scale_output_dir = scales_dir / scale_dir_name(scale)
        ensure_dir(scale_output_dir)
        save_image(samples, scale_output_dir / "grid.png", nrow=samples_per_class)
        scale_to_grid[scale] = visual_grid_tensor(samples, samples_per_class)
        print(f"[cfg-sweep] saved visual grid for scale={scale:.2f} -> {scale_output_dir / 'grid.png'}")
    generator.train(generator_was_training)

    plot_visual_appendix_figure(figures_dir, scale_to_grid, visual_scales, samples_per_class)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_name": checkpoint_path.stem,
        "eval_classifier_checkpoint": eval_checkpoint_name,
        "seed": cli_args.seed,
        "fid_scales": fid_scales,
        "visual_scales": visual_scales,
        "fid_num_gen": cli_args.fid_num_gen,
        "visual_samples_per_class": samples_per_class,
        "output_dir": str(run_dir),
        "mode": train_args.mode,
        "latent_channels": train_args.latent_channels,
        "num_workers": train_args.num_workers,
        "cfg_drop_prob": getattr(train_args, "cfg_drop_prob", 0.0),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
