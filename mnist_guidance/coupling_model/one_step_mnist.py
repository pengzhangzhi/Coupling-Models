import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import wandb

from transformer_flow import Model as ConditionalLatentFlow


LATENT_SPATIAL = 7
IMAGE_SIZE = 28
NUM_CLASSES = 10
PIXELS = IMAGE_SIZE * IMAGE_SIZE
FLOW_TOKENS = LATENT_SPATIAL * LATENT_SPATIAL


def is_cond_mode(args: argparse.Namespace) -> bool:
    return args.mode == "cond"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    return torch.device("cuda")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def divisor_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BinaryMNISTDatasetView(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        image = (image >= 0.5).float()
        sequence = image.flatten()
        return image, sequence, torch.tensor(label, dtype=torch.long)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = divisor_group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = divisor_group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class BinaryVAE(nn.Module):
    """
    Binary MNIST VAE with latent map shape [B, latent_channels, 7, 7].
    """

    def __init__(self, latent_channels: int = 16, fixed_std: float = 0.5):
        super().__init__()
        self.latent_channels = latent_channels
        self.fixed_std = fixed_std

        self.enc_in = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc_res1 = ResBlock(32, 32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_res2 = ResBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_res3 = ResBlock(128, 128)
        self.enc_out = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)

        self.dec_in = nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1)
        self.dec_res1 = ResBlock(128, 128)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        )
        self.dec_res2 = ResBlock(64, 64)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )
        self.dec_res3 = ResBlock(32, 32)
        self.dec_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_in(x)
        x = self.enc_res1(x)
        x = self.down1(x)
        x = self.enc_res2(x)
        x = self.down2(x)
        x = self.enc_res3(x)
        return self.enc_out(F.silu(x))

    def sample_latent(self, mean: torch.Tensor) -> torch.Tensor:
        return mean + self.fixed_std * torch.randn_like(mean)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.dec_in(latent)
        x = self.dec_res1(x)
        x = self.up1(x)
        x = self.dec_res2(x)
        x = self.up2(x)
        x = self.dec_res3(x)
        return self.dec_out(F.silu(x))

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.encode_mean(x)
        latent = mean if deterministic else self.sample_latent(mean)
        logits = self.decode(latent)
        return mean, latent, logits


class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, width: int, token_hidden: int, channel_hidden: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, num_tokens),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, width),
        )
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, width * 4),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_t, scale_t, shift_c, scale_c = self.film(cond).chunk(4, dim=-1)
        token_in = modulate(self.norm1(x), shift_t, scale_t).transpose(1, 2)
        x = x + self.token_mlp(token_in).transpose(1, 2)
        channel_in = modulate(self.norm2(x), shift_c, scale_c)
        x = x + self.channel_mlp(channel_in)
        return x


class ConditionalLatentToPixelMixer(nn.Module):
    """
    One-step generator mapping latent tokens [B, 49, 16] to pixel logits [B, 784].
    """

    def __init__(
        self,
        latent_dim: int = 16,
        latent_tokens: int = FLOW_TOKENS,
        num_pixels: int = PIXELS,
        width: int = 256,
        depth: int = 6,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.pixel_tokens = nn.Parameter(torch.randn(num_pixels, width) * 0.02)
        self.latent_proj = nn.Linear(latent_dim * latent_tokens, width)
        self.label_embed = nn.Embedding(num_classes, width) if num_classes > 0 else None
        self.null_label = nn.Parameter(torch.randn(1, width) * 0.02)
        self.blocks = nn.ModuleList(
            [MixerBlock(num_pixels, width, token_hidden=width, channel_hidden=width * 2, cond_dim=width) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def _label_condition(self, batch_size: int, labels: torch.Tensor | None, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None or self.label_embed is None:
            return self.null_label.expand(batch_size, -1)
        label_cond = self.label_embed(labels)
        if drop_mask is None:
            return label_cond
        drop_mask = drop_mask.unsqueeze(1).to(label_cond.dtype)
        return drop_mask * self.null_label + (1 - drop_mask) * label_cond

    def forward(
        self,
        z_tokens: torch.Tensor,
        labels: torch.Tensor | None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = z_tokens.shape[0]
        cond = self.latent_proj(z_tokens.flatten(1)) + self._label_condition(batch_size, labels, drop_mask)
        x = self.pixel_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            x = block(x, cond)
        return self.out(self.norm(x)).squeeze(-1)


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_set = BinaryMNISTDatasetView(args.data_dir, train=True, download=True)
    test_set = BinaryMNISTDatasetView(args.data_dir, train=False, download=True)
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader


def make_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BinaryVAE, ConditionalLatentFlow, ConditionalLatentToPixelMixer]:
    vae = BinaryVAE(latent_channels=args.latent_channels, fixed_std=args.fixed_std).to(device)
    flow = ConditionalLatentFlow(
        in_channels=args.latent_channels,
        img_size=LATENT_SPATIAL,
        patch_size=1,
        channels=args.flow_width,
        num_blocks=args.flow_blocks,
        layers_per_block=args.flow_layers,
        num_heads=args.flow_heads,
        num_classes=NUM_CLASSES if is_cond_mode(args) else 0,
        label_drop_prob=0.0,
    ).to(device)
    generator = ConditionalLatentToPixelMixer(
        latent_dim=args.latent_channels,
        width=args.generator_width,
        depth=args.generator_depth,
        num_classes=NUM_CLASSES if is_cond_mode(args) else 0,
    ).to(device)
    return vae, flow, generator


def latent_stats_str(latent: torch.Tensor) -> str:
    return (
        f"mean={latent.mean().item():.4f} std={latent.std().item():.4f} "
        f"min={latent.min().item():.4f} max={latent.max().item():.4f}"
    )


def log_wandb(data: dict) -> None:
    if wandb.run is not None:
        wandb.log(data)


def init_wandb(args: argparse.Namespace) -> None:
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


def checkpoint_path_for_args(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path)
    return default_checkpoint_path(args)


def generator_cfg_logits(
    generator: ConditionalLatentToPixelMixer,
    z_tokens: torch.Tensor,
    labels: torch.Tensor | None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    if labels is None or not hasattr(generator, "label_embed") or generator.label_embed is None:
        return generator(z_tokens, None)
    if cfg_scale == 1.0:
        return generator(z_tokens, labels)
    logits_u = generator(z_tokens, None)
    if cfg_scale == 0.0:
        return logits_u
    logits_c = generator(z_tokens, labels)
    return logits_u + cfg_scale * (logits_c - logits_u)


def binary_samples_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) >= 0.5).float().view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


@torch.no_grad()
def generate_binary_samples(
    generator: ConditionalLatentToPixelMixer,
    z_tokens: torch.Tensor,
    labels: torch.Tensor | None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    logits = generator_cfg_logits(generator, z_tokens, labels, cfg_scale=cfg_scale)
    return binary_samples_from_logits(logits)


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
        "config_id": args.config_id,
        "epoch": epoch,
        "mode": args.mode,
        "seed": args.seed,
        "fixed_std": args.fixed_std,
        "flow_width": args.flow_width,
        "flow_blocks": args.flow_blocks,
        "flow_layers": args.flow_layers,
        "flow_heads": args.flow_heads,
        "generator_width": args.generator_width,
        "generator_depth": args.generator_depth,
        "cfg_drop_prob": getattr(args, "cfg_drop_prob", 0.0),
        "fid_num_gen": args.fid_num_gen,
        "output_dir": str(Path(args.output_dir).resolve()),
        "checkpoint_path": str(checkpoint_path_for_args(args).resolve()),
    }
    record.update(metrics)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@torch.no_grad()
def sample_reconstructions(
    vae: BinaryVAE,
    loader: DataLoader,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    was_training = vae.training
    vae.eval()
    device = next(vae.parameters()).device
    images, _, _ = next(iter(loader))
    images = images.to(device)
    mean, _, logits = vae(images, deterministic=True)
    recon = (torch.sigmoid(logits) >= 0.5).float()
    grid = torch.cat([images[:50], recon[:50]], dim=0)
    suffix = "final" if epoch < 0 else f"epoch_{epoch:03d}"
    output_path = Path(args.output_dir) / f"recon_{args.mode}_{suffix}.png"
    save_image(grid, output_path, nrow=10)
    print(f"[samples] recon latent stats {latent_stats_str(mean)}")
    print(f"[samples] saved {output_path}")
    log_wandb(
        {
            "samples/recon_grid": wandb.Image(str(output_path)),
            "samples/latent_mean": mean.mean().item(),
            "samples/latent_std": mean.std().item(),
            "samples/epoch": max(epoch, 0),
        },
    )
    vae.train(was_training)


@torch.no_grad()
def sample_generations(
    generator: ConditionalLatentToPixelMixer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    was_training = generator.training
    generator.eval()
    device = next(generator.parameters()).device
    suffix = "final" if epoch < 0 else f"epoch_{epoch:03d}"
    step = max(epoch, 0)

    def to_wandb_image(samples: torch.Tensor) -> wandb.Image:
        return wandb.Image(make_grid(samples, nrow=10).detach().cpu())

    if is_cond_mode(args):
        labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(10)
        z_tokens = torch.randn(labels.shape[0], FLOW_TOKENS, args.latent_channels, device=device)
        samples = generate_binary_samples(generator, z_tokens, labels, cfg_scale=1.0)
        output_path = Path(args.output_dir) / f"cond_samples_{suffix}.png"
        save_image(samples, output_path, nrow=10)
        print(f"[samples] saved {output_path}")
        log_wandb(
            {
                "samples/cond_grid": to_wandb_image(samples),
                "samples/cond_labels": wandb.Histogram(labels.detach().cpu().numpy()),
                "samples/epoch": step,
            },
        )
    else:
        z_tokens = torch.randn(100, FLOW_TOKENS, args.latent_channels, device=device)
        samples = generate_binary_samples(generator, z_tokens, None, cfg_scale=1.0)
        output_path = Path(args.output_dir) / f"uncond_samples_{suffix}.png"
        save_image(samples, output_path, nrow=10)
        print(f"[samples] saved {output_path}")
        log_wandb(
            {
                "samples/uncond_grid": to_wandb_image(samples),
                "samples/epoch": step,
            },
        )

    generator.train(was_training)


def _gray_to_rgb(x_uint8: torch.Tensor) -> torch.Tensor:
    if x_uint8.ndim != 4 or x_uint8.shape[1] != 1:
        raise ValueError(f"Expected [N,1,H,W], got {tuple(x_uint8.shape)}")
    return x_uint8.repeat(1, 3, 1, 1)


def _to_uint8_0_255(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected [N,1,28,28], got {tuple(x.shape)}")
    if x.dtype == torch.uint8:
        return x
    return (x >= 0.5).to(torch.uint8) * 255


def _balanced_labels(n: int, device: torch.device) -> torch.Tensor:
    repeats = math.ceil(n / NUM_CLASSES)
    return torch.arange(NUM_CLASSES, device=device).repeat(repeats)[:n]


def default_checkpoint_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "checkpoints" / "last.pt"


def save_checkpoint(
    path: Path,
    vae: BinaryVAE,
    flow: ConditionalLatentFlow,
    generator: ConditionalLatentToPixelMixer,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "vae": vae.state_dict(),
        "flow": flow.state_dict(),
        "generator": generator.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
    }
    torch.save(payload, path)
    print(f"[checkpoint] saved {path}")


def load_checkpoint(
    path: Path,
    vae: BinaryVAE,
    flow: ConditionalLatentFlow,
    generator: ConditionalLatentToPixelMixer,
    device: torch.device,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    vae.load_state_dict(payload["vae"])
    flow.load_state_dict(payload["flow"])
    generator.load_state_dict(payload["generator"])
    print(f"[checkpoint] loaded {path}")
    return payload


def build_mnist_fid(
    data_dir: str,
    num_workers: int,
    device: torch.device,
    target_class: int | None = None,
):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError(
            "FID evaluation requires torchmetrics + torch-fidelity. "
            "Install with: pip install torchmetrics[image] torch-fidelity"
        ) from exc

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.set_dtype(torch.float64)

    real_loader = DataLoader(
        BinaryMNISTDatasetView(data_dir, train=True, download=True),
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    for real_images, _, labels in real_loader:
        if target_class is not None:
            class_mask = labels == target_class
            if not torch.any(class_mask):
                continue
            real_images = real_images[class_mask]
        real_uint8 = _to_uint8_0_255(real_images.to(device))
        fid.update(_gray_to_rgb(real_uint8), real=True)
    if fid.real_features_num_samples == 0:
        if target_class is None:
            raise RuntimeError("No real MNIST samples were loaded for FID reference.")
        raise RuntimeError(f"No real MNIST samples found for target class {target_class}.")
    return fid


def compute_fid_metric_value(fid) -> float:
    from scipy import linalg

    if fid.real_features_num_samples < 2 or fid.fake_features_num_samples < 2:
        raise RuntimeError("More than one real and fake sample is required to compute FID.")

    mean_real = (fid.real_features_sum / fid.real_features_num_samples).detach().cpu().to(torch.float64)
    mean_fake = (fid.fake_features_sum / fid.fake_features_num_samples).detach().cpu().to(torch.float64)

    cov_real_num = fid.real_features_cov_sum - fid.real_features_num_samples * torch.outer(
        fid.real_features_sum / fid.real_features_num_samples,
        fid.real_features_sum / fid.real_features_num_samples,
    )
    cov_fake_num = fid.fake_features_cov_sum - fid.fake_features_num_samples * torch.outer(
        fid.fake_features_sum / fid.fake_features_num_samples,
        fid.fake_features_sum / fid.fake_features_num_samples,
    )
    cov_real = (cov_real_num / (fid.real_features_num_samples - 1)).detach().cpu().to(torch.float64)
    cov_fake = (cov_fake_num / (fid.fake_features_num_samples - 1)).detach().cpu().to(torch.float64)

    covmean, _ = linalg.sqrtm((cov_real @ cov_fake).numpy(), disp=False)
    if not isinstance(covmean, torch.Tensor):
        covmean = torch.from_numpy(covmean.real if getattr(covmean, "dtype", None) is not None else covmean)
    covmean = covmean.to(torch.float64)
    if torch.is_complex(covmean):
        covmean = covmean.real

    diff = mean_real - mean_fake
    fid_value = diff.dot(diff) + torch.trace(cov_real + cov_fake - 2 * covmean)
    return float(fid_value.item())


@torch.no_grad()
def compute_generator_fid(
    generator: ConditionalLatentToPixelMixer,
    args: argparse.Namespace,
    device: torch.device,
    cfg_scale: float = 1.0,
    n_gen: int | None = None,
    fid=None,
) -> float:
    if fid is None:
        fid = build_mnist_fid(args.data_dir, args.num_workers, device)

    was_training = generator.training
    generator.eval()
    n_gen = args.fid_num_gen if n_gen is None else n_gen
    remaining = n_gen
    cond_labels = _balanced_labels(n_gen, device) if is_cond_mode(args) else None
    label_index = 0
    while remaining > 0:
        bsz = min(256, remaining)
        z_tokens = torch.randn(bsz, FLOW_TOKENS, args.latent_channels, device=device)
        labels = None
        if cond_labels is not None:
            labels = cond_labels[label_index : label_index + bsz]
            label_index += bsz
        fake = generate_binary_samples(generator, z_tokens, labels, cfg_scale=cfg_scale)
        fake_uint8 = _to_uint8_0_255(fake)
        fid.update(_gray_to_rgb(fake_uint8), real=False)
        remaining -= bsz

    generator.train(was_training)
    return compute_fid_metric_value(fid)


@torch.no_grad()
def evaluate_pairflow_fid(
    generator: ConditionalLatentToPixelMixer,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int | None = None,
) -> float:
    n_gen = args.fid_num_gen
    fid_value = compute_generator_fid(generator, args, device, cfg_scale=1.0, n_gen=n_gen)
    msg = (
        f"[eval] fid_pairflow_mnist_binary={fid_value:.4f} "
        f"mode={args.mode} n_gen={n_gen} reference=mnist_train steps=1"
    )
    if epoch is not None:
        msg += f" epoch={epoch}"
    print(msg)
    data = {
        "eval/fid": fid_value,
        "eval/fid_num_gen": n_gen,
    }
    if epoch is not None:
        data["eval/epoch"] = epoch
    log_wandb(data)
    append_metrics_record(
        args,
        event="eval_fid",
        epoch=epoch,
        fid=fid_value,
        test_gen_nll=getattr(args, "_latest_test_gen_nll", None),
    )
    return fid_value


@torch.no_grad()
def evaluate_test_generator_nll(
    vae: BinaryVAE,
    flow: ConditionalLatentFlow,
    generator: ConditionalLatentToPixelMixer,
    loader: DataLoader,
    args: argparse.Namespace,
    epoch: int | None = None,
) -> float:
    vae_was_training = vae.training
    flow_was_training = flow.training
    gen_was_training = generator.training
    vae.eval()
    flow.eval()
    generator.eval()

    device = next(generator.parameters()).device
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    total = 0.0
    count = 0
    for images, sequences, labels in loader:
        images = images.to(device)
        sequences = sequences.to(device)
        labels = labels.to(device)
        cond_labels = labels if is_cond_mode(args) else None
        _, latent, _ = vae(images, deterministic=False)
        z_tokens, _ = flow.forward_map(latent, context_global=cond_labels)
        gen_logits = generator(z_tokens, cond_labels)
        total += loss_fn(gen_logits, sequences).item()
        count += 1

    nll = total / max(1, count)
    msg = f"[eval] test_gen_nll={nll:.4f} mode={args.mode}"
    if epoch is not None:
        msg += f" epoch={epoch}"
    print(msg)
    data = {"eval/test_gen_nll": nll}
    if epoch is not None:
        data["eval/epoch"] = epoch
    log_wandb(data)
    args._latest_test_gen_nll = nll
    append_metrics_record(
        args,
        event="eval_test_gen_nll",
        epoch=epoch,
        fid=None,
        test_gen_nll=nll,
    )

    vae.train(vae_was_training)
    flow.train(flow_was_training)
    generator.train(gen_was_training)
    return nll


def train(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output_dir))
    device = get_device()
    set_seed(args.seed)
    train_loader, test_loader = make_dataloaders(args)
    vae, flow, generator = make_models(args, device)
    params = list(vae.parameters()) + list(flow.parameters()) + list(generator.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, len(train_loader), args.epochs)
    recon_loss_fn = nn.BCEWithLogitsLoss()
    gen_loss_fn = nn.BCEWithLogitsLoss()
    global_step = 0
    split_epoch = args.epochs // 2

    for epoch in range(1, args.epochs + 1):
        is_stage1 = epoch <= split_epoch
        phase = "stage1_vae_flow" if is_stage1 else "stage2_generator"

        for p in vae.parameters():
            p.requires_grad = is_stage1
        for p in flow.parameters():
            p.requires_grad = is_stage1
        for p in generator.parameters():
            p.requires_grad = not is_stage1

        if is_stage1:
            vae.train()
            flow.train()
            generator.eval()
        else:
            vae.eval()
            flow.eval()
            generator.train()

        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_flow = 0.0
        epoch_gen = 0.0
        epoch_acc = 0.0
        epoch_prob = 0.0

        for step, (images, sequences, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            sequences = sequences.to(device)
            labels = labels.to(device)
            cond_labels = labels if is_cond_mode(args) else None

            optimizer.zero_grad(set_to_none=True)

            if is_stage1:
                _, latent, recon_logits = vae(images, deterministic=False)
                z_tokens, logdet = flow.forward_map(latent, context_global=cond_labels)
                gen_logits = generator(z_tokens.detach(), cond_labels)

                recon_loss = recon_loss_fn(recon_logits, images)
                flow_loss = flow.get_loss(z_tokens, logdet)
                gen_loss = gen_loss_fn(gen_logits, sequences)
                total_loss = (
                    recon_loss
                    + args.lambda_flow * flow_loss
                )
            else:
                with torch.no_grad():
                    _, latent, _ = vae(images, deterministic=False)
                    z_tokens, _ = flow.forward_map(latent, context_global=cond_labels)
                drop_mask = None
                if cond_labels is not None and args.cfg_drop_prob > 0:
                    drop_mask = torch.rand(cond_labels.shape[0], device=device) < args.cfg_drop_prob
                gen_logits = generator(z_tokens, cond_labels, drop_mask=drop_mask)
                gen_loss = gen_loss_fn(gen_logits, sequences)

                recon_loss = torch.zeros((), device=device)
                flow_loss = torch.zeros((), device=device)
                total_loss = gen_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            probs = torch.sigmoid(gen_logits)
            preds = (probs >= 0.5).float()
            pixel_acc = (preds == sequences).float().mean()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_flow += flow_loss.item()
            epoch_gen += gen_loss.item()
            epoch_acc += pixel_acc.item()
            epoch_prob += probs.mean().item()

            if global_step % args.log_every == 0:
                print(
                    f"[train] phase={phase} epoch={epoch:03d} step={step:04d} total={total_loss.item():.4f} "
                    f"recon={recon_loss.item():.4f} flow={flow_loss.item():.4f} "
                    f"gen={gen_loss.item():.4f} pixel_acc={pixel_acc.item():.4f}"
                )
                log_wandb(
                    {
                        "train/phase": 1 if is_stage1 else 2,
                        "train/total_loss": total_loss.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/flow_loss": flow_loss.item(),
                        "train/gen_loss": gen_loss.item(),
                        "train/pixel_acc": pixel_acc.item(),
                        "train/mean_prob": probs.mean().item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                        "train/epoch": epoch,
                    },
                )
            global_step += 1

        num_steps = len(train_loader)
        print(
            f"[epoch] phase={phase} epoch={epoch:03d} avg_total={epoch_total / num_steps:.4f} "
            f"avg_recon={epoch_recon / num_steps:.4f} "
            f"avg_flow={epoch_flow / num_steps:.4f} "
            f"avg_gen={epoch_gen / num_steps:.4f} avg_pixel_acc={epoch_acc / num_steps:.4f}"
        )
        log_wandb(
            {
                "epoch/phase": 1 if is_stage1 else 2,
                "epoch/total_loss": epoch_total / num_steps,
                "epoch/recon_loss": epoch_recon / num_steps,
                "epoch/flow_loss": epoch_flow / num_steps,
                "epoch/gen_loss": epoch_gen / num_steps,
                "epoch/pixel_acc": epoch_acc / num_steps,
                "epoch/mean_prob": epoch_prob / num_steps,
                "epoch/epoch": epoch,
            },
        )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_reconstructions(vae, test_loader, args, epoch)
            sample_generations(generator, args, epoch)
            evaluate_test_generator_nll(vae, flow, generator, test_loader, args, epoch=epoch)
            if not args.no_fid:
                evaluate_pairflow_fid(generator, args, device, epoch=epoch)

    sample_reconstructions(vae, test_loader, args, -1)
    sample_generations(generator, args, -1)
    save_checkpoint(default_checkpoint_path(args), vae, flow, generator, args, args.epochs, global_step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a joint VAE, latent flow, and generator on binary MNIST.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/one_step_mnist")
    parser.add_argument("--mode", type=str, choices=("cond", "uncond"), default="cond")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--fixed-std", type=float, default=0.5)
    parser.add_argument("--flow-width", type=int, default=128)
    parser.add_argument("--flow-blocks", type=int, default=5)
    parser.add_argument("--flow-layers", type=int, default=5)
    parser.add_argument("--flow-heads", type=int, default=4)
    parser.add_argument("--generator-width", type=int, default=512)
    parser.add_argument("--generator-depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--lambda-gen", type=float, default=1.0)
    parser.add_argument("--wandb-project", type=str, default="one-step-mnist")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--config-id", type=str, default="default")
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--fid-num-gen", type=int, default=1000)
    parser.add_argument("--no-fid", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(Path(args.output_dir))
    init_wandb(args)
    try:
        if args.eval_only:
            device = get_device()
            set_seed(args.seed)
            _, test_loader = make_dataloaders(args)
            vae, flow, generator = make_models(args, device)
            ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else default_checkpoint_path(args)
            load_checkpoint(ckpt_path, vae, flow, generator, device)
            evaluate_test_generator_nll(vae, flow, generator, test_loader, args, epoch=None)
            if not args.no_fid:
                evaluate_pairflow_fid(generator, args, device)
        else:
            train(args)
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
