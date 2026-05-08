"""Stage A training: frozen Qwen encoder hidden states + TarFlow NF.

Data flow:
  x ∈ V^T --[Qwen(frozen)]--> h (B,T,896)
           --[+σε]--> h_noisy --[TarFlow]--> z ∈ R^{T×896}

TarFlow (ported from apple/ml-tarflow, arXiv:2412.06329):
  Stack of MetaBlocks, each a causal-transformer affine coupling over the T
  position dimension.  Alternating identity/flip permutations between blocks.
  Forward (encode, parallel):  O(num_blocks × T × layers_per_block)
  Inverse (decode, sequential): O(T × num_blocks × layers_per_block) with KV cache

Trainable parameters: TarFlow only (Qwen frozen).
"""
import os
import time
import math
import yaml
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from lightning import pytorch as pl

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ltlm_lightning import (
    FixedPathCheckpointCallback,
    TimeBudgetCallback,
    build_trainer_kwargs,
    build_wandb_logger,
    resolve_fit_checkpoint_path,
)
from prepare import (
    QWEN_MODEL, QWEN_HIDDEN, MAX_SEQ_LEN,
    PAD_TOKEN_ID, TIME_BUDGET, LTLMTrainingDataModule, build_dataloaders_for_training, get_dataloaders,
)
from runtime_paths import (
    checkpoint_root,
    configure_process_environment,
    ensure_runtime_dirs,
    sanitize_experiment_name,
    shared_hidden_stats_path,
    wandb_root,
)
from stage_a_prior_sampling import run_stage_a_prior_sampling

HIDDEN_STATS_MAX_BATCHES = 64


# ============================================================================
# TarFlow (ported from apple/ml-tarflow, arXiv:2412.06329)
# Adapted for continuous vector sequences (B, T, D) — no image patchify.
# ============================================================================

class _TarFlowAttention(nn.Module):
    """Causal self-attention used inside TarFlow MetaBlocks.

    Supports a KV-cache mode for the autoregressive inverse pass.
    """
    def __init__(self, channels: int, head_dim: int = 64):
        super().__init__()
        assert channels % head_dim == 0
        self.n_heads = channels // head_dim
        self.head_dim = head_dim
        self.norm  = nn.LayerNorm(channels)
        self.qkv   = nn.Linear(channels, channels * 3)
        self.proj  = nn.Linear(channels, channels)
        self._k_cache: list[torch.Tensor] = []
        self._v_cache: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Parallel forward (training / encoding)."""
        B, T, C = x.shape
        x_n = self.norm(x.float()).to(x.dtype)
        qkv = self.qkv(x_n).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)   # each (B, H, T, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)

    def forward_cached(self, x_i: torch.Tensor) -> torch.Tensor:
        """Single-step forward for autoregressive inverse with KV cache."""
        B, _, C = x_i.shape   # x_i: (B, 1, C)
        x_n = self.norm(x_i.float()).to(x_i.dtype)
        qkv = self.qkv(x_n).reshape(B, 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)   # each (B, H, 1, head_dim)
        self._k_cache.append(k)
        self._v_cache.append(v)
        k_all = torch.cat(self._k_cache, dim=2)
        v_all = torch.cat(self._v_cache, dim=2)
        out = F.scaled_dot_product_attention(q, k_all, v_all)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, C)
        return self.proj(out)

    def clear_cache(self):
        self._k_cache.clear()
        self._v_cache.clear()


class _TarFlowMLP(nn.Module):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1  = nn.Linear(channels, channels * expansion)
        self.fc2  = nn.Linear(channels * expansion, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x.float()).to(x.dtype))))


class _TarFlowBlock(nn.Module):
    """Single transformer block: pre-norm attention + pre-norm MLP."""
    def __init__(self, channels: int, head_dim: int = 64, expansion: int = 4):
        super().__init__()
        self.attn = _TarFlowAttention(channels, head_dim)
        self.mlp  = _TarFlowMLP(channels, expansion)

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(x, causal_mask)
        x = x + self.mlp(x)
        return x

    def forward_cached(self, x_i: torch.Tensor) -> torch.Tensor:
        x_i = x_i + self.attn.forward_cached(x_i)
        x_i = x_i + self.mlp(x_i)
        return x_i

    def clear_cache(self):
        self.attn.clear_cache()


class TarFlowMetaBlock(nn.Module):
    """One TarFlow coupling step over the T dimension.

    Forward (data→noise, parallel):
      permute(x)
      h = causal_transformer(x)         -- h[:,i] attends to x[:,0:i]
      h_shifted = cat([0, h[:,:-1]])    -- shift right: pos i uses context 0..i-1
      scale, shift = h_shifted.chunk(2)
      z_i = (x_i - shift_i) * exp(-scale_i)
      logdet = -scale.mean([1,2])       -- per-element average over T×D

    Inverse (noise→data, sequential with KV cache):
      x_0 = z_0  (identity — no context for first position)
      for i = 1..T-1:
          scale_{i-1}, shift_{i-1} = transformer(x_{i-1})  [KV-cached]
          x_i = z_i * exp(scale_{i-1}) + shift_{i-1}
    """
    def __init__(self, d: int, T: int, channels: int, n_layers: int,
                 head_dim: int = 64, expansion: int = 4, flip: bool = False):
        super().__init__()
        self.flip = flip
        self.T    = T
        self.d    = d
        self.proj_in   = nn.Linear(d, channels)
        self.pos_embed = nn.Parameter(torch.randn(T, channels) * 1e-2)
        self.blocks    = nn.ModuleList([
            _TarFlowBlock(channels, head_dim, expansion) for _ in range(n_layers)
        ])
        self.proj_out  = nn.Linear(channels, d * 2)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        # Causal mask: lower-triangular, registered as buffer
        self.register_buffer("causal_mask",
                             torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def _perm(self, x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=[1]) if self.flip else x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x    = self._perm(x)
        x_in = x.float()

        pos = self._perm(self.pos_embed.unsqueeze(0)).squeeze(0)  # (T, channels)
        h   = self.proj_in(x) + pos                               # (B, T, channels)
        for block in self.blocks:
            h = block(h, self.causal_mask)
        h = self.proj_out(h)                                      # (B, T, D*2)

        # Shift right: position i gets predicted params from context 0..i-1.
        # Keep the affine coupling math in fp32; this is the numerically fragile
        # seam where AMP can otherwise overflow the squared latent norm.
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1).float()
        scale, shift = h.chunk(2, dim=-1)                         # each (B, T, D)

        # Clamp scale to prevent exp blow-up during long training
        # scale  = scale.clamp(-5.0, 5.0)
        z      = (x_in - shift) * torch.exp(-scale)
        logdet = -scale.mean(dim=[1, 2])                          # (B,) per-element avg
        return self._perm(z), logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Autoregressive inverse with KV cache.  O(T) sequential steps."""
        z = self._perm(z).float()
        B, T, D = z.shape
        x   = z.clone()
        pos = self._perm(self.pos_embed.unsqueeze(0)).squeeze(0)  # (T, channels)

        for block in self.blocks:
            block.clear_cache()

        # Position 0: identity (no prior context → scale=0, shift=0)
        # Positions 1..T-1: recover from cached transformer
        for i in range(T - 1):
            # Feed x[:,i] through transformer (adds to KV cache)
            h_i = self.proj_in(x[:, i:i+1]) + pos[i:i+1]        # (B, 1, channels)
            for block in self.blocks:
                h_i = block.forward_cached(h_i)
            h_out            = self.proj_out(h_i).float()        # (B, 1, D*2)
            scale_i, shift_i = h_out.chunk(2, dim=-1)             # each (B, 1, D)
            # scale_i          = scale_i.clamp(-5.0, 5.0)
            x[:, i+1:i+2]   = z[:, i+1:i+2] * torch.exp(scale_i) + shift_i

        return self._perm(x)


class TarFlow(nn.Module):
    """Stack of TarFlowMetaBlocks — replaces NormalisingFlow.

    Identical interface: forward(u) → (z, logdet),  inverse(z) → u
    where u, z ∈ (B, T, D).

    Loss (per-element NLL of standard Gaussian):
      loss = 0.5 * z.pow(2).mean() - logdet.mean()
    """
    def __init__(self, d: int, T: int, num_blocks: int, layers_per_block: int,
                 channels: int, head_dim: int = 64, expansion: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            TarFlowMetaBlock(d, T, channels, layers_per_block,
                             head_dim, expansion, flip=(i % 2 == 1))
            for i in range(num_blocks)
        ])

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z      = u.float()
        logdet = torch.zeros(u.shape[0], device=u.device, dtype=torch.float32)
        for block in self.blocks:
            z, ld  = block(z)
            logdet = logdet + ld
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z.float()
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x


def compute_stage_a_flow_loss(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    """Compute the TarFlow objective in fp32.

    Under fp16 mixed precision, `z.pow(2)` can overflow once latent magnitudes
    cross ~256 even when the model is still recoverable. Casting before the
    reduction keeps the AMP path stable without disabling mixed precision for
    the rest of the step.
    """
    z_f = z.float()
    logdet_f = logdet.float()
    return 0.5 * z_f.pow(2).mean() - logdet_f.mean()



# ============================================================================
# Encoder / Decoder
# ============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm bidirectional Transformer block (used in TextDecoder)."""
    def __init__(self, d, n_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d))

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x    = x + h
        x    = x + self.ffn(self.norm2(x))
        return x


class QwenEncoder(nn.Module):
    """Frozen Qwen2.5-0.5B encoder returning raw hidden states."""
    def __init__(self, qwen_model: str = QWEN_MODEL):
        super().__init__()
        self.qwen = AutoModel.from_pretrained(qwen_model)
        for p in self.qwen.parameters():
            p.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask   = (token_ids != PAD_TOKEN_ID).long()
        with torch.no_grad():
            hidden = self.qwen(token_ids, attention_mask=mask).last_hidden_state
        return hidden.float()


def _coerce_hidden_stat(value, d: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.repeat(d)
    if tensor.ndim != 1 or tensor.shape[0] != d:
        raise ValueError(f"Stage A {name} must be scalar or shape ({d},), got {tuple(tensor.shape)}")
    return tensor


@torch.no_grad()
def compute_hidden_stats_from_loader(
    encoder: QwenEncoder,
    loader,
    device: str,
    std_floor: float,
    max_batches: int = HIDDEN_STATS_MAX_BATCHES,
) -> dict[str, torch.Tensor | int | float]:
    if std_floor <= 0:
        raise ValueError(f"Stage A std_floor must be > 0, got {std_floor}")
    if max_batches <= 0:
        raise ValueError(f"Stage A max_batches must be > 0, got {max_batches}")

    sum_hidden = None
    sumsq_hidden = None
    num_tokens = 0
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    for batch_idx, (token_ids, padding_mask) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        token_ids = token_ids.to(device)
        padding_mask = padding_mask.to(device).bool()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            hidden = encoder(token_ids).float()
        mask = padding_mask.unsqueeze(-1).to(hidden.dtype)

        batch_sum = (hidden * mask).sum(dim=(0, 1))
        batch_sumsq = (hidden.square() * mask).sum(dim=(0, 1))
        if sum_hidden is None:
            sum_hidden = batch_sum.to(torch.float64)
            sumsq_hidden = batch_sumsq.to(torch.float64)
        else:
            sum_hidden += batch_sum.to(torch.float64)
            sumsq_hidden += batch_sumsq.to(torch.float64)
        num_tokens += int(padding_mask.sum().item())

    if num_tokens == 0 or sum_hidden is None or sumsq_hidden is None:
        raise ValueError("Stage A hidden-stat computation saw zero valid tokens")

    mean = (sum_hidden / num_tokens).to(torch.float32)
    var = (sumsq_hidden / num_tokens) - sum_hidden.square() / (num_tokens ** 2)
    raw_std = var.clamp_min(0.0).sqrt().to(torch.float32)
    std = raw_std.clamp_min(std_floor)
    return {
        "mean": mean.cpu(),
        "raw_std": raw_std.cpu(),
        "std": std.cpu(),
        "num_tokens": num_tokens,
        "std_floor": float(std_floor),
    }


def resolve_hidden_stats_path(_ckpt_dir, cfg: dict, dataset: str):
    qwen_model = cfg.get("model", {}).get("qwen_model", QWEN_MODEL)
    return shared_hidden_stats_path(dataset, qwen_model)


def load_or_compute_hidden_stats(
    encoder: QwenEncoder,
    loader,
    device: str,
    stats_path,
    std_floor: float,
    force_recompute: bool = False,
    max_batches: int = HIDDEN_STATS_MAX_BATCHES,
) -> dict[str, torch.Tensor | int | float]:
    if not force_recompute and stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        required = {"mean", "raw_std", "std", "num_tokens"}
        if required <= set(stats.keys()):
            return stats

    stats = compute_hidden_stats_from_loader(
        encoder=encoder,
        loader=loader,
        device=device,
        std_floor=std_floor,
        max_batches=max_batches,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, stats_path)
    return stats


# ============================================================================
# Stage A Model
# ============================================================================

class StageAModel(nn.Module):
    """Stage A: frozen Qwen encoder hidden states + TarFlow NF."""
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        qwen_hidden = m.get("qwen_hidden", QWEN_HIDDEN)
        d = m.get("d", qwen_hidden)
        if d != qwen_hidden:
            raise ValueError(
                f"Stage A latent dimension must match Qwen hidden size for full-flow mode: d={d}, qwen_hidden={qwen_hidden}"
            )
        f = cfg["flow"]

        self.encoder = QwenEncoder(qwen_model=m.get("qwen_model", QWEN_MODEL))
        self.flow = TarFlow(
            d=d,
            T=m.get("seq_len", MAX_SEQ_LEN),
            num_blocks=f["num_blocks"],
            layers_per_block=f["layers_per_block"],
            channels=f["channels"],
            head_dim=f.get("head_dim", 64),
            expansion=f.get("expansion", 4),
        )
        self.d = d
        self.seq_len = m.get("seq_len", MAX_SEQ_LEN)
        self.register_buffer("hidden_mean", torch.zeros(d))
        self.register_buffer("hidden_std", torch.ones(d))
        self.set_hidden_stats(
            m.get("hidden_mean", m.get("global_mean", 0.0)),
            m.get("hidden_std", m.get("global_std", 1.0)),
        )
        self.sigma = m["sigma"]

    def set_hidden_stats(self, hidden_mean, hidden_std) -> None:
        mean = _coerce_hidden_stat(hidden_mean, d=self.d, name="hidden_mean")
        std = _coerce_hidden_stat(hidden_std, d=self.d, name="hidden_std")
        if torch.any(std <= 0):
            raise ValueError("Stage A hidden_std entries must all be > 0")
        self.hidden_mean.copy_(mean.to(device=self.hidden_mean.device, dtype=self.hidden_mean.dtype))
        self.hidden_std.copy_(std.to(device=self.hidden_std.device, dtype=self.hidden_std.dtype))

    def encode_raw(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(token_ids)

    def normalize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return (hidden - self.hidden_mean) / self.hidden_std

    def unnormalize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.hidden_std + self.hidden_mean

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.normalize_hidden(self.encode_raw(token_ids))

    def decode_latents(self, z: torch.Tensor, *, unnormalize: bool = True) -> torch.Tensor:
        hidden = self.flow.inverse(z)
        return self.unnormalize_hidden(hidden) if unnormalize else hidden

    def forward(self, token_ids: torch.Tensor, deterministic: bool = False):
        h = self.encode(token_ids)
        h_noisy = h if deterministic else h + self.sigma * torch.randn_like(h)
        z, logdet = self.flow(h_noisy)
        return h, z, logdet


def log_stage_a_prior_samples(
    stage_a: StageAModel,
    qwen_lm,
    tokenizer,
    device: str,
    step: int,
    num_samples: int,
    seq_len: int,
    z_scale: float,
    temperature: float,
    top_p: float,
) -> None:
    metrics, texts = run_stage_a_prior_sampling(
        stage_a=stage_a,
        qwen_lm=qwen_lm,
        tokenizer=tokenizer,
        device=device,
        num_samples=num_samples,
        seq_len=seq_len,
        z_scale=z_scale,
        temperature=temperature,
        top_p=top_p,
    )
    table = wandb.Table(columns=["step", "sample_id", "text"])
    for sample_id, text in enumerate(texts):
        table.add_data(step + 1, sample_id, text)
    wandb.log(
        {
            "prior/z_cycle_mse": metrics["z_cycle_mse"],
            "prior/z_cycle_mae": metrics["z_cycle_mae"],
            "prior/z_cycle_cos": metrics["z_cycle_cos"],
            "prior/hidden_mean": metrics["hidden_mean"],
            "prior/hidden_std": metrics["hidden_std"],
            "prior/hidden_norm": metrics["hidden_norm"],
            "samples/text": table,
        },
        step=step + 1,
    )


# ============================================================================
# Training
# ============================================================================

def _resolve_experiment_name(cfg: dict, config_path: str, experiment_name: str | None) -> str:
    if experiment_name:
        return sanitize_experiment_name(experiment_name)
    wb_name = cfg.get("wandb", {}).get("run_name")
    if wb_name:
        return sanitize_experiment_name(str(wb_name))
    return sanitize_experiment_name(os.path.splitext(os.path.basename(config_path))[0])


def build_stage_a_runtime(
    cfg: dict,
    dataset: str,
    config_path: str,
    experiment_name: str | None = None,
) -> dict:
    exp_name = _resolve_experiment_name(cfg, config_path, experiment_name)
    ckpt_dir = checkpoint_root() / "stage_a" / f"v3_{dataset}" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint_dir": ckpt_dir,
        "checkpoint_path": ckpt_dir / "checkpoint.ckpt",
        "experiment_name": exp_name,
        "stats_path": resolve_hidden_stats_path(ckpt_dir, cfg, dataset),
        "config_path": config_path,
    }


class StageALightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg: dict,
        *,
        dataset: str,
        checkpoint_path,
        stats_path,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.stats_path = stats_path
        self.model = StageAModel(cfg)
        self._hidden_stats_ready = False
        self._qwen_lm = None
        self._tokenizer = None
        tc = cfg["training"]
        self.prior_samples = int(tc.get("prior_samples", 8))
        self.prior_z_scale = float(tc.get("prior_z_scale", 1.0))
        self.prior_temperature = float(tc.get("prior_temperature", 0.7))
        self.prior_top_p = float(tc.get("prior_top_p", 0.95))
        self.log_every = int(tc.get("log_every", 100))
        self.eval_every = int(tc.get("eval_every", 2000))
        self.save_hyperparameters(ignore=["cfg", "checkpoint_path", "stats_path"])

    @property
    def legacy_global_step(self) -> int:
        return int(self.global_step)

    def setup(self, stage: str) -> None:
        if stage not in ("fit", "validate") or self._hidden_stats_ready:
            return
        norm_cfg = self.cfg.get("normalization", {})
        if self.trainer.is_global_zero:
            encoder_device = str(next(self.model.encoder.parameters()).device)
            subset_fraction = float(self.cfg["training"].get("train_subset_fraction", 1.0))
            subset_seed = int(self.cfg["training"].get("train_subset_seed", 0))
            stats_loader, _ = build_dataloaders_for_training(
                self.cfg["training"]["batch_size"],
                dataset=self.dataset,
                num_workers=0,
                train_subset_fraction=subset_fraction,
                train_subset_seed=subset_seed,
            )
            stats = load_or_compute_hidden_stats(
                encoder=self.model.encoder,
                loader=DataLoader(
                    stats_loader.dataset,
                    batch_size=self.cfg["training"]["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                ),
                device=encoder_device,
                stats_path=self.stats_path,
                std_floor=float(norm_cfg.get("std_floor", 1.0e-3)),
                force_recompute=bool(norm_cfg.get("recompute", False)),
            )
            self.model.set_hidden_stats(stats["mean"], stats["std"])
        self.trainer.strategy.barrier("stage_a_hidden_stats")
        stats = torch.load(self.stats_path, map_location="cpu", weights_only=False)
        self.model.set_hidden_stats(stats["mean"], stats["std"])
        self.cfg["model"]["hidden_mean"] = stats["mean"].tolist()
        self.cfg["model"]["hidden_std"] = stats["std"].tolist()
        self._hidden_stats_ready = True

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and hasattr(self.logger, "experiment"):
            self.logger.experiment.config.update(
                {
                    "dataset": self.dataset,
                    "device": str(self.device),
                    "normalization_stats_path": str(self.stats_path),
                    "prior_samples": self.prior_samples,
                    "prior_z_scale": self.prior_z_scale,
                    "prior_temperature": self.prior_temperature,
                    "prior_top_p": self.prior_top_p,
                },
                allow_val_change=True,
            )

    def configure_optimizers(self):
        tc = self.cfg["training"]
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
            betas=(0.9, 0.999),
        )
        warmup_steps = tc["warmup_steps"]
        max_steps = tc["max_steps"]

        def lr_lambda(step: int):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        token_ids, _ = batch
        h, z, logdet = self.model(token_ids, deterministic=False)
        loss_flow = compute_stage_a_flow_loss(z, logdet)
        if not torch.isfinite(loss_flow) or loss_flow.item() > 1e6:
            z_f = z.detach().float()
            logdet_f = logdet.detach().float()
            raise RuntimeError(
                "Stage A encountered invalid loss: "
                f"loss={loss_flow.item():.4e} "
                f"z_abs_max={z_f.abs().max().item():.4e} "
                f"z_std={z_f.std().item():.4e} "
                f"logdet_min={logdet_f.min().item():.4e} "
                f"logdet_max={logdet_f.max().item():.4e}"
            )
        self.log("train/loss_flow", loss_flow, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/hidden_mean", h.mean(), on_step=True, sync_dist=True)
        self.log("train/hidden_std", h.std(), on_step=True, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, sync_dist=False)
        return loss_flow

    def _ensure_sampling_models(self) -> None:
        if self._qwen_lm is None:
            self._qwen_lm = AutoModelForCausalLM.from_pretrained(QWEN_MODEL).to(self.device)
            self._qwen_lm.eval()
            for param in self._qwen_lm.parameters():
                param.requires_grad_(False)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _barrier(self, name: str) -> None:
        if getattr(self.trainer, "world_size", 1) > 1:
            self.trainer.strategy.barrier(name)

    @torch.no_grad()
    def _run_eval_snapshot(self, limit_examples: int) -> None:
        if not self.trainer.is_global_zero:
            return
        _, val_loader = build_dataloaders_for_training(
            self.cfg["training"]["batch_size"],
            dataset=self.dataset,
            num_workers=0,
        )
        self.model.eval()
        zs, norms = [], []
        seen = 0
        for tids, _ in val_loader:
            tids = tids.to(self.device)
            u_m = self.model.encode(tids)
            zb, _ = self.model.flow(u_m)
            norms.append(zb.norm(dim=-1).flatten().cpu())
            zs.append(zb.reshape(-1, self.model.d).cpu())
            seen += tids.shape[0]
            if seen >= limit_examples:
                break
        if not zs:
            return
        all_z = torch.cat(zs)
        all_n = torch.cat(norms)
        z_var = all_z.var(dim=0)
        z_kl = 0.5 * (all_z.pow(2).mean() + z_var.mean() - z_var.log().mean() - 1)
        wandb.log(
            {
                "eval/z_mean_norm": all_n.mean().item(),
                "eval/z_std": all_z.std(dim=0).mean().item(),
                "eval/z_kl_approx": z_kl.item(),
            },
            step=self.legacy_global_step,
        )
        self._ensure_sampling_models()
        log_stage_a_prior_samples(
            stage_a=self.model,
            qwen_lm=self._qwen_lm,
            tokenizer=self._tokenizer,
            device=str(self.device),
            step=self.legacy_global_step - 1,
            num_samples=self.prior_samples,
            seq_len=self.model.seq_len,
            z_scale=self.prior_z_scale,
            temperature=self.prior_temperature,
            top_p=self.prior_top_p,
        )
        self.model.train()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.legacy_global_step > 0 and self.eval_every > 0 and self.legacy_global_step % self.eval_every == 0:
            self._barrier("stage_a_periodic_eval_start")
            self._run_eval_snapshot(limit_examples=512)
            self._barrier("stage_a_periodic_eval_end")

    def on_train_end(self) -> None:
        self._barrier("stage_a_train_end_eval_start")
        self._run_eval_snapshot(limit_examples=5000)
        self._barrier("stage_a_train_end_eval_end")
        if self.trainer.is_global_zero and self.cfg.get("wandb", {}).get("save_artifact", True):
            artifact = wandb.Artifact("stage_a_checkpoint", type="model")
            artifact.add_file(str(self.checkpoint_path))
            wandb.log_artifact(artifact)

    def build_legacy_checkpoint(self, trainer: pl.Trainer) -> dict:
        return {
            "step": self.legacy_global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": trainer.optimizers[0].state_dict(),
            "config": self.cfg,
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ltlm_stage_a_cfg"] = self.cfg
        state_dict = checkpoint.get("state_dict", {})
        for key in [name for name in state_dict if name.startswith("_qwen_lm.")]:
            state_dict.pop(key, None)


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="lm1b", choices=["lm1b", "owt"])
    parser.add_argument("--config",      default="configs/stage_a.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ensure_runtime_dirs()
    configure_process_environment()
    tc = cfg["training"]
    subset_fraction = float(tc.get("train_subset_fraction", 1.0))
    subset_seed = int(tc.get("train_subset_seed", 0))
    experiment_name = os.getenv("EXPERIMENT_NAME")
    runtime = build_stage_a_runtime(
        cfg,
        dataset=args.dataset,
        config_path=args.config,
        experiment_name=experiment_name,
    )
    print(f"[train_stage_a] shared hidden stats: {runtime['stats_path']}")
    logger = build_wandb_logger(cfg, wandb_root(), run_name=runtime["experiment_name"])
    datamodule = LTLMTrainingDataModule(
        batch_size=tc["batch_size"],
        dataset=args.dataset,
        num_workers=4,
        train_subset_fraction=subset_fraction,
        train_subset_seed=subset_seed,
        train_shuffle_seed=int(tc.get("train_shuffle_seed", 0)),
    )

    module = StageALightningModule(
        cfg,
        dataset=args.dataset,
        checkpoint_path=runtime["checkpoint_path"],
        stats_path=runtime["stats_path"],
    )
    fit_checkpoint_path = resolve_fit_checkpoint_path(
        runtime["checkpoint_path"],
        required_datamodule_key=LTLMTrainingDataModule.__name__,
    )
    if fit_checkpoint_path is not None:
        print(f"[train_stage_a] Auto-resuming from {fit_checkpoint_path}")
    callbacks = [
        TimeBudgetCallback(TIME_BUDGET),
        FixedPathCheckpointCallback(
            runtime["checkpoint_path"],
            every_n_train_steps=25_000,
        ),
    ]
    trainer = pl.Trainer(
        **build_trainer_kwargs(
            cfg,
            max_steps=tc["max_steps"],
            logger=logger,
            callbacks=callbacks,
            use_distributed_sampler=False,
        ),
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=fit_checkpoint_path)
    if trainer.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    train()
