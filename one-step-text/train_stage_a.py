"""Stage A training: learned token encoder + TarFlow NF + linear LM head.

Data flow:
  x ∈ V^T --[Embedding]--> h (B,T,896) --[Linear]--> u_mean (B,T,256)
           --[+σε]-->  u  --[TarFlow]--> z ∈ R^{T×256}
  u --[Linear LM head]--> logits ∈ R^{T×V}

TarFlow (ported from apple/ml-tarflow, arXiv:2412.06329):
  Stack of MetaBlocks, each a causal-transformer affine coupling over the T
  position dimension.  Alternating identity/flip permutations between blocks.
  Forward (encode, parallel):  O(num_blocks × T × layers_per_block)
  Inverse (decode, sequential): O(T × num_blocks × layers_per_block) with KV cache

Trainable parameters: token embedding, encoder projection, TarFlow blocks, LM head.
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

from prepare import (
    QWEN_VOCAB_SIZE, MAX_SEQ_LEN,
    PAD_TOKEN_ID, get_dataloaders,
)
from runtime_paths import (
    configure_process_environment,
    ensure_runtime_dirs,
    resolve_checkpoint_path,
    wandb_root,
)


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
        x_in = x

        pos = self._perm(self.pos_embed.unsqueeze(0)).squeeze(0)  # (T, channels)
        h   = self.proj_in(x) + pos                               # (B, T, channels)
        for block in self.blocks:
            h = block(h, self.causal_mask)
        h = self.proj_out(h)                                      # (B, T, D*2)

        # Shift right: position i gets predicted params from context 0..i-1
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        scale, shift = h.chunk(2, dim=-1)                         # each (B, T, D)

        # Clamp scale to prevent exp blow-up during long training
        scale  = scale.clamp(-5.0, 5.0)
        z      = (x_in - shift) * torch.exp(-scale)
        logdet = -scale.mean(dim=[1, 2])                          # (B,) per-element avg
        return self._perm(z), logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Autoregressive inverse with KV cache.  O(T) sequential steps."""
        z = self._perm(z)
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
            h_out            = self.proj_out(h_i)                 # (B, 1, D*2)
            scale_i, shift_i = h_out.chunk(2, dim=-1)             # each (B, 1, D)
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
        z      = u
        logdet = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for block in self.blocks:
            z, ld  = block(z)
            logdet = logdet + ld
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x



# ============================================================================
# Encoder / Decoder
# ============================================================================

class LearnedTokenEncoder(nn.Module):
    """Learned token embedding plus linear projection.

    Input:  token_ids (B, T)
    Output: u_mean    (B, T, repr_dim)
    """
    def __init__(self, vocab_size: int, embed_dim: int, repr_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.proj  = nn.Linear(embed_dim, repr_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(token_ids))


class LinearTextDecoder(nn.Module):
    """Single LM-head-style projection: u → logits ∈ R^{T×V}."""
    def __init__(self, d: int, vocab_size: int):
        super().__init__()
        self.out_proj = nn.Linear(d, vocab_size)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.out_proj(u)


# ============================================================================
# Stage A Model
# ============================================================================

class StageAModel(nn.Module):
    """Stage A: learned token encoder + TarFlow NF + linear LM head."""
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = m["d"]
        f = cfg["flow"]

        self.encoder = LearnedTokenEncoder(
            vocab_size=QWEN_VOCAB_SIZE,
            embed_dim=m.get("embed_dim", 896),
            repr_dim=d,
        )
        self.decoder = LinearTextDecoder(d=d, vocab_size=QWEN_VOCAB_SIZE)
        self.flow = TarFlow(
            d=d,
            T=m.get("seq_len", MAX_SEQ_LEN),
            num_blocks=f["num_blocks"],
            layers_per_block=f["layers_per_block"],
            channels=f["channels"],
            head_dim=f.get("head_dim", 64),
            expansion=f.get("expansion", 4),
        )
        self.sigma = m["sigma"]
        self.d     = d

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(token_ids)

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.decoder(u)

    def forward(self, token_ids: torch.Tensor, deterministic: bool = False):
        u_mean  = self.encode(token_ids)
        u       = u_mean if deterministic else u_mean + self.sigma * torch.randn_like(u_mean)
        z, logdet = self.flow(u)
        logits    = self.decode(u)
        return logits, z, logdet


# ============================================================================
# Training
# ============================================================================

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--dataset",     default="lm1b", choices=["lm1b", "owt"])
    parser.add_argument("--config",      default="configs/stage_a.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tc = cfg["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_stage_a] device={device}  config={args.config}")

    ensure_runtime_dirs()
    configure_process_environment()

    wb = cfg.get("wandb", {})
    wandb_dir = wandb_root()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        entity=wb.get("entity", None),
        project=wb.get("project", "latent-transport-lm"),
        name=wb.get("run_name", None),
        tags=wb.get("tags", []),
        config=cfg,
        dir=str(wandb_dir),
    )
    wandb.config.update({"device": device, "dataset": args.dataset}, allow_val_change=True)

    train_loader, val_loader = get_dataloaders(tc["batch_size"], dataset=args.dataset)

    print(f"[train_stage_a] Initializing learned Stage A model ...")
    model = StageAModel(cfg).to(device)

    checkpoints_cfg = cfg.get("checkpoints", {})
    if "save_path" not in checkpoints_cfg:
        raise KeyError("configs must define checkpoints.save_path for Stage A")
    ckpt_path = resolve_checkpoint_path(checkpoints_cfg["save_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train_stage_a] checkpoint save path: {ckpt_path}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[train_stage_a] params: {n_trainable:,} trainable / {n_total:,} total")
    wandb.config.update({"n_params_trainable": n_trainable, "n_params_total": n_total},
                        allow_val_change=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tc["lr"], weight_decay=tc["weight_decay"], betas=(0.9, 0.999),
    )
    warmup_steps = tc["warmup_steps"]
    max_steps    = tc["max_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    start_step = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_step = ckpt["step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[train_stage_a] Resumed from step {start_step}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )

    use_amp   = tc.get("mixed_precision", True) and device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    lf_start = tc["lambda_flow_start"]
    lf_end   = tc["lambda_flow_end"]
    lf_frac  = tc["lambda_flow_anneal_frac"]

    def get_lambda_flow(step):
        anneal_steps = int(lf_frac * max_steps)
        if step >= anneal_steps:
            return lf_end
        return lf_start + (lf_end - lf_start) * step / anneal_steps

    t0      = time.time()
    step    = start_step
    running = dict(recon=0., flow=0., total=0.)
    log_every  = tc.get("log_every", 100)
    eval_every = tc.get("eval_every", 2000)

    print(f"[train_stage_a] Starting training (max_steps={max_steps})")

    while step < max_steps:
        for token_ids, padding_mask in train_loader:
            if step >= max_steps:
                break

            token_ids    = token_ids.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, z, logdet = model(token_ids, deterministic=False)

                B, T, V = logits.shape
                loss_recon_all = F.cross_entropy(
                    logits.reshape(-1, V), token_ids.reshape(-1), reduction="none",
                ).reshape(B, T)
                mask_sum   = padding_mask.sum()
                loss_recon = (loss_recon_all * padding_mask).sum() / mask_sum.clamp(min=1)

                # Pure NLL: TarFlow logdet is already per-element mean over T×D
                loss_flow = 0.5 * z.pow(2).mean() - logdet.mean()

                lam_flow = get_lambda_flow(step)
                loss     = loss_recon + lam_flow * loss_flow

            if not torch.isfinite(loss) or loss.item() > 1e6:
                print(f"[train_stage_a] WARNING: bad loss={loss.item():.4e} at step {step+1}, skipping")
                optimizer.zero_grad()
                scheduler.step()
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], tc["grad_clip"]
            )
            optimizer.step()
            scheduler.step()

            running["recon"] += loss_recon.item()
            running["flow"]  += loss_flow.item()
            running["total"] += loss.item()

            if (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                avg = {k: v / log_every for k, v in running.items()}
                lr  = optimizer.param_groups[0]["lr"]
                print(
                    f"[train_stage_a] step={step+1:6d}  elapsed={elapsed:.0f}s  "
                    f"recon={avg['recon']:.4f}  flow={avg['flow']:.4f}  "
                    f"total={avg['total']:.4f}  λ_flow={lam_flow:.3f}  lr={lr:.2e}"
                )
                wandb.log({
                    "train/loss_recon":  avg["recon"],
                    "train/loss_flow":   avg["flow"],
                    "train/loss_total":  avg["total"],
                    "train/lambda_flow": lam_flow,
                    "train/lr":          lr,
                    "train/elapsed_s":   elapsed,
                }, step=step + 1)
                for k in running:
                    running[k] = 0.0

            if (step + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    zs, norms = [], []
                    seen = 0
                    for tids, _ in val_loader:
                        tids = tids.to(device)
                        u_m  = model.encode(tids)
                        zb, _ = model.flow(u_m)
                        norms.append(zb.norm(dim=-1).flatten().cpu())
                        zs.append(zb.reshape(-1, model.d).cpu())
                        seen += tids.shape[0]
                        if seen >= 512:
                            break
                    all_z = torch.cat(zs)
                    all_n = torch.cat(norms)
                z_kl = 0.5 * (all_z.pow(2).mean() + all_z.var(dim=0).mean()
                               - all_z.var(dim=0).log().mean() - 1)
                wandb.log({
                    "eval/z_mean_norm": all_n.mean().item(),
                    "eval/z_std":       all_z.std(dim=0).mean().item(),
                    "eval/z_kl_approx": z_kl.item(),
                }, step=step + 1)
                model.train()

            if (step + 1) % 25_000 == 0:
                torch.save({
                    "step": step + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                }, ckpt_path)
                print(f"[train_stage_a] Checkpoint saved at step {step+1}")

            step += 1

        if step >= max_steps:
            break

    elapsed = time.time() - t0
    print(f"\n[train_stage_a] Finished: {step} steps in {elapsed:.1f}s")

    torch.save({
        "step": step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"[train_stage_a] Saved {ckpt_path}")

    # ── Final val eval ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for tids, pmask in val_loader:
            tids, pmask = tids.to(device), pmask.to(device)
            u_m    = model.encode(tids)
            logits = model.decode(u_m)
            preds  = logits.argmax(dim=-1)
            correct += ((preds == tids) & pmask).sum().item()
            total   += pmask.sum().item()
            if total >= 10_000:
                break
        recon_acc = correct / max(total, 1)

        zs, norms = [], []
        seen = 0
        for tids, _ in val_loader:
            tids = tids.to(device)
            u_m  = model.encode(tids)
            zb, _ = model.flow(u_m)
            norms.append(zb.norm(dim=-1).flatten().cpu())
            zs.append(zb.reshape(-1, model.d).cpu())
            seen += tids.shape[0]
            if seen >= 5_000:
                break
        all_z = torch.cat(zs)
        all_n = torch.cat(norms)
        z_kl  = 0.5 * (all_z.pow(2).mean() + all_z.var(dim=0).mean()
                        - all_z.var(dim=0).log().mean() - 1)

    print(f"\nrecon_acc={recon_acc:.4f}")
    print(f"z_mean_norm={all_n.mean().item():.4f}  (ideal: sqrt(D)={model.d**0.5:.1f})")
    print(f"z_std={all_z.std(dim=0).mean().item():.4f}  (ideal: 1.0)")
    print(f"z_kl_approx={z_kl.item():.4f}  (ideal: 0.0)")

    wandb.log({
        "eval/recon_acc":   recon_acc,
        "eval/z_mean_norm": all_n.mean().item(),
        "eval/z_std":       all_z.std(dim=0).mean().item(),
        "eval/z_kl_approx": z_kl.item(),
    }, step=step)

    if wb.get("save_artifact", True):
        artifact = wandb.Artifact("stage_a_checkpoint", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train()
