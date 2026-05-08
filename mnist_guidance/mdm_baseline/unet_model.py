from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from one_step_mnist import NUM_CLASSES, ensure_dir
from mdm_baseline.model import MASK_TOKEN, VOCAB_SIZE


UNET_IMAGE_SIZE = 32
UNET_PIXELS = UNET_IMAGE_SIZE * UNET_IMAGE_SIZE


def divisor_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = divisor_group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = divisor_group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_channels * 2))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        while scale.ndim < h.ndim:
            scale = scale[..., None]
            shift = shift[..., None]
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.norm = divisor_group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        h = self.norm(x).reshape(bsz, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        head_dim = channels // self.num_heads
        q = q.reshape(bsz, self.num_heads, head_dim, height * width).transpose(2, 3)
        k = k.reshape(bsz, self.num_heads, head_dim, height * width)
        v = v.reshape(bsz, self.num_heads, head_dim, height * width).transpose(2, 3)
        attn = torch.softmax((q @ k) * (head_dim**-0.5), dim=-1)
        h = (attn @ v).transpose(2, 3).reshape(bsz, channels, height * width)
        h = self.proj(h).reshape(bsz, channels, height, width)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNetMDM(nn.Module):
    """Class-conditional 32x32 token-embedding U-Net for binary masked diffusion."""

    def __init__(
        self,
        image_size: int = UNET_IMAGE_SIZE,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8, 16),
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_pixels = image_size * image_size
        self.base_channels = base_channels
        self.token_embed = nn.Embedding(VOCAB_SIZE, base_channels)
        self.label_embed = nn.Embedding(num_classes, base_channels * 4)
        self.null_label = nn.Parameter(torch.randn(1, base_channels * 4) * 0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        self.input_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels: list[int] = [base_channels]
        channels = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(channels, out_channels, base_channels * 4, dropout))
                channels = out_channels
                if resolution in attention_resolutions:
                    blocks.append(AttentionBlock(channels, num_heads))
                skip_channels.append(channels)
            self.down_blocks.append(blocks)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(channels))
                resolution //= 2
                skip_channels.append(channels)

        self.mid1 = ResBlock(channels, channels, base_channels * 4, dropout)
        self.mid_attn = AttentionBlock(channels, num_heads)
        self.mid2 = ResBlock(channels, channels, base_channels * 4, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock(channels + skip_channels.pop(), out_channels, base_channels * 4, dropout))
                channels = out_channels
                if resolution in attention_resolutions:
                    blocks.append(AttentionBlock(channels, num_heads))
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(channels))
                resolution *= 2

        self.out = nn.Sequential(
            divisor_group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def _label_condition(
        self,
        batch_size: int,
        labels: torch.Tensor | None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is None:
            return self.null_label.expand(batch_size, -1)
        cond = self.label_embed(labels)
        if drop_mask is None:
            return cond
        drop_mask = drop_mask.unsqueeze(1).to(cond.dtype)
        return drop_mask * self.null_label + (1 - drop_mask) * cond

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor | float,
        labels: torch.Tensor | None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.num_pixels:
            raise ValueError(f"Expected tokens [B,{self.num_pixels}], got {tuple(tokens.shape)}")
        batch_size = tokens.shape[0]
        if not torch.is_tensor(t):
            t = torch.full((batch_size,), float(t), device=tokens.device)
        elif t.ndim == 0:
            t = t.expand(batch_size)
        else:
            t = t.to(tokens.device)

        cond = self.time_embed(timestep_embedding(t, self.base_channels)) + self._label_condition(batch_size, labels, drop_mask)
        x = tokens.view(batch_size, self.image_size, self.image_size)
        h = self.token_embed(x).permute(0, 3, 1, 2)
        h = self.input_conv(h)
        skips = [h]
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, cond) if isinstance(block, ResBlock) else block(h)
                if isinstance(block, ResBlock):
                    skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)

        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, cond)
                else:
                    h = block(h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        return self.out(h).flatten(1)


def parse_channel_mult(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def parse_resolutions(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def make_unet_mdm_from_args(args: argparse.Namespace, device: torch.device) -> ConditionalUNetMDM:
    return ConditionalUNetMDM(
        base_channels=args.base_channels,
        channel_mult=parse_channel_mult(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=parse_resolutions(args.attention_resolutions),
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)


def save_unet_checkpoint(
    path: Path,
    model: ConditionalUNetMDM,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    extra: dict | None = None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "model_type": "unet_mdm",
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[checkpoint] saved {path}")


def load_unet_checkpoint(
    path: Path,
    model: ConditionalUNetMDM,
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
