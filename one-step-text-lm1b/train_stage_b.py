"""Stage B training: Transformer generator with KD from frozen Qwen2.5-0.5B.

Data flow:
  token_ids → Stage A (frozen): encode → dequant → flow → z  (B, T, latent_dim)
  z         → Generator:        z → gen_logits              (B, T, V)
  token_ids → Qwen LM (frozen): token_ids → qwen_logits     (B, T, V)
  Loss = CE(gen_logits, token_ids) + λ_kd · KL(gen/T ‖ qwen/T) · T²

Z mixing: with probability z_gauss_prob, replace real Stage A z with pure
N(0, sample_z_scale²·I). Forces generator to learn the prior→text mapping
directly, preventing prior gap from widening during training.
"""
import argparse
import copy
import math
import os
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning import pytorch as pl

from transformers import AutoModelForCausalLM, AutoTokenizer

from ltlm_lightning import (
    FixedPathCheckpointCallback,
    TimeBudgetCallback,
    build_trainer_kwargs,
    build_wandb_logger,
    resolve_fit_checkpoint_path,
)
from prepare import (
    QWEN_MODEL, QWEN_VOCAB_SIZE, MAX_SEQ_LEN, QWEN_HIDDEN,
    PAD_TOKEN_ID, TIME_BUDGET, LTLMTrainingDataModule, build_dataloaders_for_training,
)
from runtime_paths import (
    checkpoint_root,
    configure_process_environment,
    ensure_runtime_dirs,
    resolve_checkpoint_path,
    sanitize_experiment_name,
    wandb_root,
)
from train_stage_a import StageAModel


# ============================================================================
# Generator Architecture
# ============================================================================

class GenTransformerBlock(nn.Module):
    """Transformer block with gated residuals and normalized cross-attention.

    Three sub-layers, each with a zero-initialized per-channel gate:
      1. Self-attention     — standard pre-norm, gated residual
      2. Cross-attention    — Q=x, K/V=LayerNorm(z_kv), gated residual
      3. FFN                — standard pre-norm, gated residual

    All gates start at zero → block is identity at init, forcing the model
    to learn a direct z→tokens mapping before residual paths open up.
    This prevents cross-attention from learning large z-invariant constants
    early in training (the failure mode diagnosed in the v1 checkpoint).

    AdaLN removed: diagnostic showed it learned harmful modulations (+0.48 CE
    improvement when zeroed). z conditioning now comes purely through the
    cross-attention K/V path.
    """
    def __init__(self, width: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.kv_norm = nn.LayerNorm(width)   # normalises cross-attn K/V — prevents value explosion
        self.cross_attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, width),
        )
        # Zero-init per-channel gates: each residual starts at 0 contribution
        self.gate_sa  = nn.Parameter(torch.zeros(width))
        self.gate_ca  = nn.Parameter(torch.zeros(width))
        self.gate_ffn = nn.Parameter(torch.zeros(width))

    def forward(self, x, z_kv):
        # Self-attention — gated residual
        x_n = self.norm1(x)
        x = x + self.gate_sa.tanh() * self.self_attn(x_n, x_n, x_n, need_weights=False)[0]
        # Cross-attention — K/V are layer-normalised to prevent norm explosion
        z_n = self.kv_norm(z_kv)
        x = x + self.gate_ca.tanh() * self.cross_attn(
            self.norm2(x), z_n, z_n, need_weights=False
        )[0]
        # FFN — gated residual
        x = x + self.gate_ffn.tanh() * self.ffn(self.norm3(x))
        return x


class TextGeneratorTransformer(nn.Module):
    """One-shot Transformer generator: z (B,T,d) → (logits (B,T,V), z_hat (B,T,d)).

    Architecture (v2):
      x    = z_proj(z) + position_bias        — init from z
      z_kv = z_proj(z)                        — shared K/V for all cross-attention blocks
      for each block:
          x = block(x, z_kv)                  — gated SA + gated CA + gated FFN
      h      = norm(x)
      logits = out_proj(h)
      z_hat  = z_recon(h)                     — auxiliary z-reconstruction head

    Zero-init gates ensure the model starts as a direct linear z→tokens mapping
    and opens residual paths only as gradient flow justifies them.
    z_hat enables an MSE loss that directly penalises z-invariant behaviour.
    """
    def __init__(self, seq_len: int, latent_dim: int, vocab_size: int,
                 width: int, depth: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.seq_len  = seq_len
        self.latent_dim = latent_dim
        self.z_proj        = nn.Linear(latent_dim, width)
        self.position_bias = nn.Parameter(torch.zeros(seq_len, width))
        self.blocks = nn.ModuleList([
            GenTransformerBlock(width, num_heads, ffn_dim)
            for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, vocab_size)
        self.z_recon  = nn.Linear(width, latent_dim)   # auxiliary z-reconstruction head

    def forward(self, z):
        z_kv = self.z_proj(z)                              # (B, T, W)
        x    = z_kv + self.position_bias.unsqueeze(0)      # (B, T, W)
        for block in self.blocks:
            x = block(x, z_kv)
        h      = self.norm(x)
        logits = self.out_proj(h)
        z_hat  = self.z_recon(h)                           # (B, T, repr_dim)
        return logits, z_hat


# ============================================================================
# EMA
# ============================================================================

class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def to(self, device: torch.device | str) -> None:
        self.shadow.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)


# ============================================================================
# Sampling
# ============================================================================

@torch.no_grad()
def generate_samples(generator: nn.Module, num_samples: int, device: str,
                     latent_dim: int,
                     z_scale: float = 1.0, temperature: float = 0.0,
                     top_p: float = 1.0) -> torch.Tensor:
    """Sample z ~ N(0, z_scale²·I) → token ids.

    temperature=0 → argmax; temperature>0 → categorical with optional top-p.
    """
    generator.eval()
    z = torch.randn(num_samples, generator.seq_len, latent_dim, device=device) * z_scale
    logits, _ = generator(z)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        remove = (cumprobs - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        N, T, V = sorted_probs.shape
        sampled = torch.multinomial(sorted_probs.view(-1, V), num_samples=1).view(N, T)
        tokens  = sorted_idx.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
    else:
        N, T, V = probs.shape
        tokens = torch.multinomial(probs.view(-1, V), num_samples=1).view(N, T)
    return tokens


def decode_tokens(token_ids: torch.Tensor) -> list[str]:
    """Decode Qwen token IDs to text strings."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    texts = []
    for ids in token_ids.cpu().tolist():
        ids = [i for i in ids if i != PAD_TOKEN_ID]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


# ============================================================================
# Training Loop
# ============================================================================

def _save_checkpoint(path, step, generator, optimizer, ema, cfg):
    data = {
        "step": step,
        "generator_state_dict": generator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    if ema is not None:
        data["ema_state_dict"] = ema.shadow.state_dict()
    torch.save(data, path)


def _resolve_stage_a_checkpoint_for_run(
    configured_path: str,
    *,
    experiment_name: str,
) -> Path:
    direct_path = resolve_checkpoint_path(configured_path)
    if direct_path.exists():
        return direct_path

    candidate = Path(configured_path).expanduser()
    if candidate.name == "checkpoint.ckpt":
        inferred = direct_path.parent / experiment_name / direct_path.name
        if inferred.exists():
            return inferred

    raise FileNotFoundError(
        f"Stage A checkpoint not found. Checked {direct_path}"
        + (
            f" and inferred same-experiment path {inferred}"
            if candidate.name == "checkpoint.ckpt"
            else ""
        )
    )


def _load_stage_a_from_checkpoint(path) -> tuple[StageAModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "ltlm_stage_a_cfg" in checkpoint and "state_dict" in checkpoint:
        stage_a_cfg = checkpoint["ltlm_stage_a_cfg"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    elif "config" in checkpoint and "model_state_dict" in checkpoint:
        stage_a_cfg = checkpoint["config"]
        model_state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError(
            f"Unsupported Stage A checkpoint format at {path}: expected Lightning checkpoint "
            "with ltlm_stage_a_cfg/state_dict or legacy config/model_state_dict"
        )
    stage_a = StageAModel(stage_a_cfg)
    stage_a.load_state_dict(model_state_dict, strict=False)
    return stage_a, stage_a_cfg


def _resolve_experiment_name(cfg: dict, config_path: str, experiment_name: str | None) -> str:
    if experiment_name:
        return sanitize_experiment_name(experiment_name)
    wb_name = cfg.get("wandb", {}).get("run_name")
    if wb_name:
        return sanitize_experiment_name(str(wb_name))
    return sanitize_experiment_name(os.path.splitext(os.path.basename(config_path))[0])


def build_stage_b_runtime(
    cfg: dict,
    dataset: str,
    config_path: str,
    experiment_name: str | None = None,
) -> dict:
    exp_name = _resolve_experiment_name(cfg, config_path, experiment_name)
    ckpt_dir = checkpoint_root() / "stage_b" / f"v3_{dataset}" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint_dir": ckpt_dir,
        "checkpoint_path": ckpt_dir / "checkpoint.ckpt",
        "stage_a_checkpoint_path": _resolve_stage_a_checkpoint_for_run(
            cfg["training"]["stage_a_checkpoint"],
            experiment_name=exp_name,
        ),
        "experiment_name": exp_name,
        "config_path": config_path,
    }


class StageBLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg: dict,
        *,
        dataset: str,
        checkpoint_path,
        stage_a_checkpoint_path,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.stage_a_checkpoint_path = stage_a_checkpoint_path
        tc = cfg["training"]
        gc = cfg["generator"]

        self.stage_a, stage_a_cfg = _load_stage_a_from_checkpoint(self.stage_a_checkpoint_path)
        self.stage_a.eval()
        for p in self.stage_a.parameters():
            p.requires_grad_(False)

        self.qwen_lm = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, torch_dtype=torch.bfloat16)
        self.qwen_lm.eval()
        for p in self.qwen_lm.parameters():
            p.requires_grad_(False)

        self.sigma = stage_a_cfg["model"]["sigma"]
        self.latent_dim = stage_a_cfg["model"].get("d", stage_a_cfg["model"].get("qwen_hidden", QWEN_HIDDEN))
        self.generator = TextGeneratorTransformer(
            seq_len=MAX_SEQ_LEN,
            latent_dim=self.latent_dim,
            vocab_size=QWEN_VOCAB_SIZE,
            width=gc["width"],
            depth=gc["depth"],
            num_heads=gc["num_heads"],
            ffn_dim=gc["ffn_dim"],
        )
        ema_decay = tc.get("ema_decay", 0.999)
        self.ema = EMA(self.generator, decay=ema_decay) if ema_decay else None

        self.label_smoothing = tc.get("label_smoothing", 0.05)
        self.lambda_kd = tc.get("lambda_kd", 1.0)
        self.lambda_recon = tc.get("lambda_recon", 0.1)
        self.lambda_nce = tc.get("lambda_nce", 0.0)
        self.nce_temperature = tc.get("nce_temperature", 0.1)
        self.nce_warmup_steps = tc.get("nce_warmup_steps", 2000)
        self.kd_temperature = tc.get("kd_temperature", 2.0)
        self.sample_every = tc.get("sample_every", 5000)
        self.num_samples_gen = tc.get("num_samples", 8)
        self.sample_z_scale = tc.get("sample_z_scale", 1.0)
        self.z_gauss_prob = tc.get("z_gauss_prob", 0.3)
        self._tokenizer = None
        self.save_hyperparameters(ignore=["cfg", "checkpoint_path", "stage_a_checkpoint_path"])

    @property
    def legacy_global_step(self) -> int:
        return int(self.global_step)

    def on_fit_start(self) -> None:
        if self.ema is not None:
            self.ema.to(next(self.generator.parameters()).device)
        if self.trainer.is_global_zero and hasattr(self.logger, "experiment"):
            self.logger.experiment.config.update(
                {
                    "dataset": self.dataset,
                    "device": str(self.device),
                    "n_params_generator": sum(p.numel() for p in self.generator.parameters()),
                },
                allow_val_change=True,
            )

    def configure_optimizers(self):
        tc = self.cfg["training"]
        optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
            betas=(0.9, 0.999),
        )
        warmup_steps = tc["warmup_steps"]
        restart_period = tc.get("restart_period", 50000)

        def lr_lambda(step: int):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = step - warmup_steps
            progress = (t % restart_period) / restart_period
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.generator)

    def training_step(self, batch, batch_idx):
        token_ids, padding_mask = batch
        with torch.no_grad():
            u_mean = self.stage_a.encode(token_ids)
            u = u_mean + self.sigma * torch.randn_like(u_mean)
            z_real, _ = self.stage_a.flow(u)
            if torch.rand(1, device=self.device).item() < self.z_gauss_prob:
                z = torch.randn_like(z_real) * self.sample_z_scale
            else:
                aug_scale = 0.80 + 0.30 * torch.rand(1, device=self.device).item()
                z = z_real * aug_scale

            attn_mask = (token_ids != PAD_TOKEN_ID).long()
            qwen_logits = self.qwen_lm(token_ids, attention_mask=attn_mask).logits.float()

        gen_logits, z_hat = self.generator(z)
        bsz, seq_len, vocab = gen_logits.shape
        loss_ce_all = F.cross_entropy(
            gen_logits.reshape(-1, vocab),
            token_ids.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(bsz, seq_len)
        mask_sum = padding_mask.sum()
        loss_ce = (loss_ce_all * padding_mask).sum() / mask_sum.clamp(min=1)

        kd_mask = padding_mask[:, 1:]
        gen_log_probs = F.log_softmax(gen_logits[:, 1:].float() / self.kd_temperature, dim=-1)
        qwen_probs = F.softmax(qwen_logits[:, :-1] / self.kd_temperature, dim=-1).detach()
        kd_per_pos = F.kl_div(
            gen_log_probs.reshape(-1, vocab),
            qwen_probs.reshape(-1, vocab),
            reduction="none",
        ).sum(dim=-1).reshape(bsz, seq_len - 1)
        kd_mask_sum = kd_mask.sum()
        loss_kd = (kd_per_pos * kd_mask).sum() / kd_mask_sum.clamp(min=1)
        loss_kd = loss_kd * (self.kd_temperature ** 2)

        loss_recon = F.mse_loss(z_hat, z.detach())

        use_nce = self.lambda_nce > 0 and self.legacy_global_step >= self.nce_warmup_steps
        if use_nce:
            gen_logits_nce, _ = self.generator(z_real.detach())
            log_probs_nce = F.log_softmax(gen_logits_nce.float(), dim=-1)
            scores = torch.zeros(bsz, bsz, device=self.device)
            for j in range(bsz):
                tgt_j = token_ids[j].unsqueeze(0).expand(bsz, -1)
                tok_lp = log_probs_nce.gather(-1, tgt_j.unsqueeze(-1)).squeeze(-1)
                msk_j = padding_mask[j].unsqueeze(0).expand(bsz, -1).float()
                scores[:, j] = (tok_lp * msk_j).sum(-1) / msk_j.sum(-1).clamp(min=1)
            logits_nce = scores / self.nce_temperature
            labels_nce = torch.arange(bsz, device=self.device)
            loss_nce = F.cross_entropy(logits_nce, labels_nce)
        else:
            loss_nce = torch.tensor(0.0, device=self.device)

        loss = loss_ce + self.lambda_kd * loss_kd + self.lambda_recon * loss_recon + self.lambda_nce * loss_nce
        preds = gen_logits.argmax(dim=-1)
        correct = ((preds == token_ids) & padding_mask).sum()
        acc = correct / mask_sum.clamp(min=1)

        self.log("train/loss_ce", loss_ce, on_step=True, sync_dist=True)
        self.log("train/loss_kd", loss_kd, on_step=True, sync_dist=True)
        self.log("train/loss_recon", loss_recon, on_step=True, sync_dist=True)
        self.log("train/loss_nce", loss_nce, on_step=True, sync_dist=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, sync_dist=False)
        return loss

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _barrier(self, name: str) -> None:
        if getattr(self.trainer, "world_size", 1) > 1:
            self.trainer.strategy.barrier(name)

    @torch.no_grad()
    def _log_samples(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self._ensure_tokenizer()
        eval_gen = self.ema.shadow if self.ema is not None else self.generator
        sample_ids = generate_samples(
            eval_gen,
            self.num_samples_gen,
            str(self.device),
            latent_dim=self.latent_dim,
            z_scale=self.sample_z_scale,
            temperature=0.9,
            top_p=0.95,
        )
        texts = decode_tokens(sample_ids)
        table = wandb.Table(columns=["step", "sample_id", "text"])
        for i, text in enumerate(texts):
            table.add_data(self.legacy_global_step, i, text)
        wandb.log({"samples/text": table}, step=self.legacy_global_step)

    @torch.no_grad()
    def _run_eval_snapshot(self) -> None:
        if not self.trainer.is_global_zero:
            return
        _, val_loader = build_dataloaders_for_training(
            self.cfg["training"]["batch_size"],
            dataset=self.dataset,
            num_workers=0,
        )
        eval_gen = self.ema.shadow if self.ema is not None else self.generator
        eval_gen.eval()
        total_correct = total_tokens = 0
        total_ce = 0.0
        n_batches = 0
        for token_ids, padding_mask in val_loader:
            token_ids = token_ids.to(self.device)
            padding_mask = padding_mask.to(self.device)
            u_mean = self.stage_a.encode(token_ids)
            z, _ = self.stage_a.flow(u_mean)
            logits, _ = eval_gen(z)
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == token_ids) & padding_mask).sum().item()
            total_tokens += padding_mask.sum().item()
            bsz, seq_len, vocab = logits.shape
            ce_all = F.cross_entropy(logits.reshape(-1, vocab), token_ids.reshape(-1), reduction="none").reshape(bsz, seq_len)
            total_ce += (ce_all * padding_mask).sum().item()
            n_batches += 1
            if n_batches >= 40:
                break
        gen_acc = total_correct / max(total_tokens, 1)
        gen_ce = total_ce / max(total_tokens, 1)
        wandb.log({"eval/gen_acc": gen_acc, "eval/gen_ce": gen_ce}, step=self.legacy_global_step)
        eval_gen.train()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.legacy_global_step > 0 and self.sample_every > 0 and self.legacy_global_step % self.sample_every == 0:
            self._barrier("stage_b_sample_logging_start")
            self._log_samples()
            self._barrier("stage_b_sample_logging_end")

    def on_train_end(self) -> None:
        self._barrier("stage_b_train_end_eval_start")
        self._run_eval_snapshot()
        self._log_samples()
        self._barrier("stage_b_train_end_eval_end")
        if self.trainer.is_global_zero and self.cfg.get("wandb", {}).get("save_artifact", False):
            artifact = wandb.Artifact("stage_b_checkpoint", type="model")
            artifact.add_file(str(self.checkpoint_path))
            wandb.log_artifact(artifact)

    def build_legacy_checkpoint(self, trainer: pl.Trainer) -> dict:
        data = {
            "step": self.legacy_global_step,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": trainer.optimizers[0].state_dict(),
            "config": self.cfg,
        }
        if self.ema is not None:
            data["ema_state_dict"] = self.ema.shadow.state_dict()
        return data


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lm1b", choices=["lm1b", "owt"])
    parser.add_argument("--config", default="configs/stage_b.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tc = cfg["training"]
    ensure_runtime_dirs()
    configure_process_environment()
    experiment_name = os.getenv("EXPERIMENT_NAME")
    runtime = build_stage_b_runtime(
        cfg,
        dataset=args.dataset,
        config_path=args.config,
        experiment_name=experiment_name,
    )
    datamodule = LTLMTrainingDataModule(
        batch_size=tc["batch_size"],
        dataset=args.dataset,
        num_workers=4,
        train_subset_fraction=float(tc.get("train_subset_fraction", 1.0)),
        train_subset_seed=int(tc.get("train_subset_seed", 0)),
        train_shuffle_seed=int(tc.get("train_shuffle_seed", 0)),
    )

    module = StageBLightningModule(
        cfg,
        dataset=args.dataset,
        checkpoint_path=runtime["checkpoint_path"],
        stage_a_checkpoint_path=runtime["stage_a_checkpoint_path"],
    )
    logger = build_wandb_logger(cfg, wandb_root(), run_name=runtime["experiment_name"])
    fit_checkpoint_path = resolve_fit_checkpoint_path(
        runtime["checkpoint_path"],
        required_datamodule_key=LTLMTrainingDataModule.__name__,
    )
    if fit_checkpoint_path is not None:
        print(f"[train_stage_b] Auto-resuming from {fit_checkpoint_path}")
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
