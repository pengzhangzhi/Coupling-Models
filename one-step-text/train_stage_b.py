"""Stage B training: latent generator/denoiser with KD from frozen Qwen2.5-0.5B.

Data flow:
  token_ids → Stage A (frozen): encode → dequant → flow → z  (B, T, 256)
  z         → Generator:        z or (masked token_ids, z) → gen_logits
  token_ids → Qwen LM (frozen): token_ids → qwen_logits     (B, T, V)
  Loss = CE(gen_logits, token_ids) + λ_kd · KL(gen/T ‖ qwen/T) · T²

Z mixing: with probability z_gauss_prob, replace real Stage A z with pure
N(0, sample_z_scale²·I). Forces generator to learn the prior→text mapping
directly, preventing prior gap from widening during training.
For generator.arch=latent_mdm, z mixing is disabled and training uses masked
token corruption conditioned on the paired Stage A latent.
"""
import argparse
import copy
import math
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Callable, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import (
    QWEN_MODEL, QWEN_VOCAB_SIZE, MAX_SEQ_LEN, REPR_DIM,
    PAD_TOKEN_ID, get_dataloaders,
)
from runtime_paths import (
    configure_process_environment,
    ensure_runtime_dirs,
    resolve_checkpoint_path,
    wandb_root,
)
from train_stage_a import StageAModel


# ============================================================================
# Generator Architecture
# ============================================================================

class SwiGLU(nn.Module):
    def __init__(self, width: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, ffn_dim)
        self.up_proj = nn.Linear(width, ffn_dim)
        self.down_proj = nn.Linear(ffn_dim, width)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GenTransformerBlock(nn.Module):
    """Bidirectional Transformer++ block: RMSNorm attention + RMSNorm SwiGLU."""
    def __init__(self, width: int, num_heads: int, ffn_dim: int):
        super().__init__()
        assert width % num_heads == 0, "width must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.attn_norm = nn.RMSNorm(width)
        self.qkv_proj = nn.Linear(width, width * 3)
        self.out_proj = nn.Linear(width, width)
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_dim)

    def forward(self, x):
        B, T, C = x.shape
        x_n = self.attn_norm(x.float()).to(x.dtype)
        qkv = self.qkv_proj(x_n).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TextGeneratorTransformer(nn.Module):
    """One-shot Transformer generator: z (B,T,d) -> logits (B,T,V).

    Architecture (v3):
      x = z_proj(z) + learned absolute position bias
      for each block:
          x = x + bidirectional_attention(RMSNorm(x))
          x = x + SwiGLU(RMSNorm(x))
      h      = final RMSNorm(x)
      logits = out_proj(h)

    Stage B receives latent z rather than discrete input tokens, so z_proj
    fills the token-embedding role in a standard transformer stack.
    """
    def __init__(self, seq_len: int, repr_dim: int, vocab_size: int,
                 width: int, depth: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.seq_len  = seq_len
        self.repr_dim = repr_dim
        self.z_proj        = nn.Linear(repr_dim, width)
        self.position_bias = nn.Parameter(torch.zeros(seq_len, width))
        self.blocks = nn.ModuleList([
            GenTransformerBlock(width, num_heads, ffn_dim)
            for _ in range(depth)
        ])
        self.norm     = nn.RMSNorm(width)
        self.out_proj = nn.Linear(width, vocab_size)

    def forward(self, z):
        x = self.z_proj(z) + self.position_bias.unsqueeze(0)  # (B, T, W)
        for block in self.blocks:
            x = block(x)
        h      = self.norm(x)
        logits = self.out_proj(h)
        return logits


class LatentMaskedDenoiser(nn.Module):
    """Latent-conditioned masked denoiser: (xt, z) -> x0 logits.

    xt contains normal Qwen token IDs plus a configured input-only mask token.
    The model is not time-conditioned; the mask pattern is the only corruption
    signal besides the fixed latent z.
    """
    def __init__(self, seq_len: int, repr_dim: int, vocab_size: int,
                 width: int, depth: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.repr_dim = repr_dim
        self.token_embed = nn.Embedding(vocab_size, width)
        self.z_proj = nn.Linear(repr_dim, width)
        self.position_bias = nn.Parameter(torch.zeros(seq_len, width))
        self.blocks = nn.ModuleList([
            GenTransformerBlock(width, num_heads, ffn_dim)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(width)
        self.out_proj = nn.Linear(width, vocab_size)

    def forward(self, xt, z):
        x = self.token_embed(xt) + self.z_proj(z) + self.position_bias.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.norm(x))


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

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)


# ============================================================================
# Sampling
# ============================================================================

def get_mask_token_id(tokenizer: AutoTokenizer) -> int:
    """Ensure Qwen tokenizer exposes an input-only mask token."""
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    if tokenizer.mask_token_id is None:
        raise ValueError("Failed to configure tokenizer.mask_token_id")
    if tokenizer.mask_token_id >= QWEN_VOCAB_SIZE:
        raise ValueError(
            f"mask_token_id={tokenizer.mask_token_id} is outside QWEN_VOCAB_SIZE={QWEN_VOCAB_SIZE}"
        )
    return tokenizer.mask_token_id


def sample_mdm_mask(
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    full_mask_prob: float = 0.5,
) -> torch.Tensor:
    """Sample MDM masks, mixing fully masked rows with Bernoulli(t) rows."""
    B, T = token_ids.shape
    t = torch.rand(B, 1, device=token_ids.device)
    t = (t+0.1).clamp(0.1, 0.1)
    full_rows = torch.rand(B, 1, device=token_ids.device) < full_mask_prob
    t = torch.where(full_rows, torch.ones_like(t), t)
    mask = (torch.rand(B, T, device=token_ids.device) < t) & padding_mask.bool()
    empty_rows = (~mask).all(dim=1)
    if empty_rows.any():
        for row in empty_rows.nonzero(as_tuple=True)[0].tolist():
            valid = padding_mask[row].nonzero(as_tuple=True)[0]
            if valid.numel() > 0:
                j = valid[torch.randint(valid.numel(), (1,), device=token_ids.device)]
                mask[row, j] = True
    return mask


@torch.no_grad()
def generate_samples(generator: nn.Module, num_samples: int, device: str,
                     z_scale: float = 1.0, temperature: float = 0.0,
                     top_p: float = 1.0) -> torch.Tensor:
    """Sample z ~ N(0, z_scale²·I) → token ids.

    temperature=0 → argmax; temperature>0 → categorical with optional top-p.
    """
    generator.eval()
    z = torch.randn(num_samples, MAX_SEQ_LEN, REPR_DIM, device=device) * z_scale
    logits = generator(z)
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


def topk_masking(scores: torch.Tensor, cutoff_len: torch.Tensor,
                 mode: str = "lowest") -> torch.Tensor:
    sorted_scores = scores.sort(dim=-1, descending=(mode == "highest")).values
    cutoff_len = cutoff_len.clamp(min=0, max=scores.size(-1) - 1)
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_len)
    return (scores >= cutoff) if mode == "highest" else (scores < cutoff)


def sample_categorical(
    logits: torch.Tensor, temperature: float = 1.0, noise_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.to(torch.float64)
    if temperature > 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = logits / temperature + noise_scale * gumbel_noise
    log_probs = logits.log_softmax(dim=-1)
    scores, tokens = log_probs.max(dim=-1)
    return tokens, scores.to(logits.dtype), logits.to(logits.dtype)


@torch.no_grad()
def p2_sampling(
    xt: torch.Tensor,
    generator: nn.Module,
    z: torch.Tensor,
    mask_id: int,
    num_steps: int,
    tau: float = 1.0,
    kappa_fn: Callable[[float], float] = lambda t: t,
    eta: float = 1.0,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1 for P2 sampling")
    dt = 1 / num_steps
    fix_mask = (xt != mask_id)
    x0 = xt.clone()

    for i in range(1, num_steps + 1):
        t = i * dt
        kappa_t = kappa_fn(t)

        logits = torch.nan_to_num(generator(xt, z).float(), nan=0.0, posinf=50.0, neginf=-50.0)
        logits[..., mask_id] = float("-inf")
        last_mask = (xt == mask_id)
        unmask_t = ~last_mask & ~fix_mask

        x0, score, _ = sample_categorical(logits, temperature=tau)
        score = score.masked_fill(fix_mask, float("inf"))
        score[unmask_t] *= eta

        num_to_mask = ((~fix_mask).sum(dim=1, keepdim=True).float() * (1 - kappa_t)).long()
        to_mask = topk_masking(score, num_to_mask, mode="lowest")

        xt[to_mask] = mask_id
        mask_2_x0 = last_mask & ~to_mask
        xt[mask_2_x0] = x0[mask_2_x0]

    xt[xt == mask_id] = x0[xt == mask_id]
    return xt


@torch.no_grad()
def generate_mdm_samples(generator: nn.Module, num_samples: int, device: str,
                         mask_token_id: int, z_scale: float = 1.0,
                         num_steps: int = 8, tau: float = 1.0,
                         eta: float = 1.0) -> torch.Tensor:
    generator.eval()
    z = torch.randn(num_samples, MAX_SEQ_LEN, REPR_DIM, device=device) * z_scale
    xt = torch.full((num_samples, MAX_SEQ_LEN), mask_token_id, device=device, dtype=torch.long)
    return p2_sampling(xt, generator, z, mask_token_id, num_steps, tau=tau, eta=eta)


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


def _run_mdm_eval_sweep(
    generator: nn.Module,
    cfg: dict,
    device: str,
    train_step: int,
    *,
    mdm_steps_list: Sequence[int],
    mdm_tau: float,
    mdm_eta: float,
    z_scale: float,
    num_samples: int,
    gpt2_model: str,
    batch_size: int,
) -> list[dict]:
    """Run the same Stage B generation eval used by eval.py for MDM sweeps."""
    from eval import evaluate, print_table

    generator.eval()
    if device == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for n_steps in mdm_steps_list:
        rows.append(evaluate(
            generator, device,
            num_samples=num_samples,
            temperature=0.0,
            top_p=1.0,
            z_scale=z_scale,
            gpt2_model=gpt2_model,
            batch_size=batch_size,
            cfg=cfg,
            mdm_steps=n_steps,
            mdm_tau=mdm_tau,
            mdm_eta=mdm_eta,
        ))

    print_table(rows, "latent_mdm")

    table = wandb.Table(columns=[
        "train_step", "mdm_steps", "mdm_tau", "mdm_eta", "z_scale",
        "num_samples", "gen_ppl", "entropy",
    ])
    log_data = {}
    for row in rows:
        n_steps = row["mdm_steps"]
        table.add_data(
            train_step, n_steps, row["mdm_tau"], row["mdm_eta"], row["z_scale"],
            row["num_samples"], row["gen_ppl"], row["entropy"],
        )
        log_data[f"eval/mdm_steps_{n_steps}/gen_ppl"] = row["gen_ppl"]
        log_data[f"eval/mdm_steps_{n_steps}/entropy"] = row["entropy"]

    log_data["eval/mdm_sweep"] = table
    wandb.log(log_data, step=train_step)
    return rows


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", default="lm1b", choices=["lm1b", "owt"])
    parser.add_argument("--config", default="configs/stage_b.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--reset_optimizer", action="store_true",
                        help="On resume: load model weights only, discard optimizer state. "
                             "Use when resuming at a cosine-restart boundary to avoid "
                             "Adam second-moment explosion.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tc  = cfg["training"]
    gc  = cfg["generator"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_stage_b] device={device}")
    ensure_runtime_dirs()
    configure_process_environment()
    checkpoints_cfg = cfg.get("checkpoints", {})
    if "stage_a_path" not in checkpoints_cfg:
        raise KeyError("configs must define checkpoints.stage_a_path for Stage B")
    if "save_path" not in checkpoints_cfg:
        raise KeyError("configs must define checkpoints.save_path for Stage B")

    # --- Load frozen Stage A ---
    ckpt_path_a = resolve_checkpoint_path(checkpoints_cfg["stage_a_path"])
    print(f"[train_stage_b] stage_a checkpoint path: {ckpt_path_a}")
    print(f"[train_stage_b] Loading Stage A from {ckpt_path_a}")
    ckpt_a = torch.load(ckpt_path_a, map_location=device, weights_only=False)
    stage_a_cfg = ckpt_a["config"]

    stage_a = StageAModel(stage_a_cfg).to(device)
    stage_a.load_state_dict(ckpt_a["model_state_dict"], strict=False)
    if hasattr(stage_a.encoder, "proj") and hasattr(stage_a.encoder.proj, "fitted"):
        stage_a.encoder.proj.fitted.fill_(True)
    stage_a.eval()
    for p in stage_a.parameters():
        p.requires_grad_(False)
    print(f"[train_stage_b] Stage A loaded (step={ckpt_a['step']}, frozen)")

    sigma = stage_a_cfg["model"]["sigma"]

    # --- Load frozen Qwen LM teacher ---
    print(f"[train_stage_b] Loading Qwen LM teacher ({QWEN_MODEL}) ...")
    qwen_lm = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.bfloat16,
    ).to(device)
    qwen_lm.eval()
    for p in qwen_lm.parameters():
        p.requires_grad_(False)
    print("[train_stage_b] Qwen LM teacher loaded and frozen")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    mask_token_id = get_mask_token_id(tokenizer)
    print(f"[train_stage_b] tokenizer.mask_token_id={mask_token_id}")

    # --- W&B ---
    # Use a project-local dir so wandb temp/staging files don't land in /tmp
    # (a shared, size-limited filesystem on compute nodes).
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

    train_loader, val_loader = get_dataloaders(tc["batch_size"], dataset=args.dataset)

    # --- Generator ---
    arch = gc.get("arch", "transformer")
    if arch == "transformer":
        generator = TextGeneratorTransformer(
            seq_len=MAX_SEQ_LEN, repr_dim=REPR_DIM, vocab_size=QWEN_VOCAB_SIZE,
            width=gc["width"], depth=gc["depth"],
            num_heads=gc["num_heads"], ffn_dim=gc["ffn_dim"],
        ).to(device)
    elif arch == "latent_mdm":
        generator = LatentMaskedDenoiser(
            seq_len=MAX_SEQ_LEN, repr_dim=REPR_DIM, vocab_size=QWEN_VOCAB_SIZE,
            width=gc["width"], depth=gc["depth"],
            num_heads=gc["num_heads"], ffn_dim=gc["ffn_dim"],
        ).to(device)
        cfg.setdefault("training", {})["mask_token_id"] = mask_token_id
    else:
        raise ValueError(f"Unknown generator.arch={arch!r}")

    n_params = sum(p.numel() for p in generator.parameters())
    print(f"[train_stage_b] generator params: {n_params:,}")
    wandb.config.update({"n_params_generator": n_params, "device": device},
                        allow_val_change=True)

    ema_decay = tc.get("ema_decay", 0.999)
    ema = EMA(generator, decay=ema_decay) if ema_decay else None

    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=tc["lr"],
        weight_decay=tc["weight_decay"], betas=(0.9, 0.999),
    )
    warmup_steps = tc["warmup_steps"]
    max_steps    = tc["max_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # Monotonic cosine decay after warmup, matching the Stage A and
        # teacher schedules. LR decays from the peak value to 10% of peak by
        # max_steps.
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # --- Resume ---
    start_step  = 0
    ckpt_path_b = resolve_checkpoint_path(checkpoints_cfg["save_path"])
    ckpt_path_b.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train_stage_b] generator checkpoint save path: {ckpt_path_b}")
    if args.resume and ckpt_path_b.exists():
        ckpt_b = torch.load(ckpt_path_b, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt_b["generator_state_dict"], strict=False)
        if ema is not None and "ema_state_dict" in ckpt_b:
            ema.shadow.load_state_dict(ckpt_b["ema_state_dict"], strict=False)
        optimizer = torch.optim.AdamW(
            generator.parameters(), lr=tc["lr"],
            weight_decay=tc["weight_decay"], betas=(0.9, 0.999),
        )
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt_b["optimizer_state_dict"])
        else:
            # LambdaLR requires 'initial_lr' in param_groups when last_epoch >= 0
            for group in optimizer.param_groups:
                group["initial_lr"] = group["lr"]
        start_step = ckpt_b["step"]
        print(f"[train_stage_b] Resumed from step {start_step}"
              + (" (optimizer state reset)" if args.reset_optimizer else ""))
    elif args.resume:
        print("[train_stage_b] --resume set but no checkpoint found; starting fresh")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )

    use_amp   = tc.get("mixed_precision", False) and device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    label_smoothing   = tc.get("label_smoothing", 0.05)
    lambda_kd         = tc.get("lambda_kd", 1.0)
    kd_temperature    = tc.get("kd_temperature", 2.0)
    sample_every      = tc.get("sample_every", 5000)
    num_samples_gen   = tc.get("num_samples", 8)
    final_num_samples = tc.get("final_num_samples", 16)
    eval_batches      = tc.get("eval_batches", 40)
    sample_z_scale    = tc.get("sample_z_scale", 1.00)
    z_gauss_prob      = tc.get("z_gauss_prob", 0.3)
    log_every         = tc.get("log_every", 100)
    mdm_steps         = tc.get("mdm_steps", 8)
    mdm_tau           = tc.get("mdm_tau", 1.0)
    mdm_eta           = tc.get("mdm_eta", 1.0)
    mdm_full_mask_prob = tc.get("mdm_full_mask_prob", 0.5)
    eval_sweep_every  = tc.get("eval_sweep_every", 0)
    eval_mdm_steps    = tc.get("eval_mdm_steps", [2, 4, 8])
    eval_mdm_tau      = tc.get("eval_mdm_tau", 0.4)
    eval_mdm_eta      = tc.get("eval_mdm_eta", 1.0)
    eval_num_samples  = tc.get("eval_num_samples", 1024)
    eval_gpt2_model   = tc.get("eval_gpt2_model", "gpt2-large")
    eval_batch_size   = tc.get("eval_batch_size", 64)

    t0 = time.time()
    step = start_step
    running_ce = running_kd = running_acc = 0.0

    print(f"[train_stage_b] Starting training (max_steps={max_steps})")
    if eval_sweep_every and arch != "latent_mdm":
        raise ValueError("training.eval_sweep_every is only supported for generator.arch=latent_mdm")
    if eval_sweep_every:
        print(
            "[train_stage_b] MDM eval sweep enabled: "
            f"every={eval_sweep_every}, steps={eval_mdm_steps}, "
            f"tau={eval_mdm_tau}, eta={eval_mdm_eta}, "
            f"num_samples={eval_num_samples}, gpt2_model={eval_gpt2_model}"
        )

    while step < max_steps:
        for token_ids, padding_mask in train_loader:
            if step >= max_steps:
                break

            token_ids    = token_ids.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                u_mean = stage_a.encode(token_ids)
                u      = u_mean + sigma * torch.randn_like(u_mean)
                z_real, _ = stage_a.flow(u)
                if arch == "latent_mdm":
                    z = z_real
                elif torch.rand(1).item() < z_gauss_prob:
                    z = torch.randn_like(z_real) * sample_z_scale
                else:
                    aug_scale = 0.80 + 0.30 * torch.rand(1, device=device).item()
                    z = z_real * aug_scale

            with torch.no_grad():
                attn_mask   = (token_ids != PAD_TOKEN_ID).long()
                qwen_logits = qwen_lm(
                    token_ids, attention_mask=attn_mask,
                ).logits.float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if arch == "latent_mdm":
                    mdm_mask = sample_mdm_mask(
                        token_ids, padding_mask,
                        full_mask_prob=mdm_full_mask_prob,
                    )
                    xt = token_ids.masked_fill(mdm_mask, mask_token_id)
                    gen_logits = generator(xt, z)
                    ce_mask = mdm_mask
                else:
                    gen_logits = generator(z)
                    ce_mask = padding_mask

                B, T, V           = gen_logits.shape

                loss_ce_all = F.cross_entropy(
                    gen_logits.reshape(-1, V), token_ids.reshape(-1),
                    reduction="none", label_smoothing=label_smoothing,
                ).reshape(B, T)
                ce_mask_sum = ce_mask.sum()
                loss_ce  = (loss_ce_all * ce_mask).sum() / ce_mask_sum.clamp(min=1)

                kd_mask        = padding_mask[:, 1:]
                gen_log_probs  = F.log_softmax(gen_logits[:, 1:].float() / kd_temperature, dim=-1)
                qwen_probs     = F.softmax(qwen_logits[:, :-1] / kd_temperature, dim=-1).detach()
                kd_per_pos = F.kl_div(
                    gen_log_probs.reshape(-1, V),
                    qwen_probs.reshape(-1, V),
                    reduction="none",
                ).sum(dim=-1).reshape(B, T - 1)
                kd_mask_sum = kd_mask.sum()
                loss_kd = (kd_per_pos * kd_mask).sum() / kd_mask_sum.clamp(min=1)
                loss_kd = loss_kd * (kd_temperature ** 2)

                loss = loss_ce + lambda_kd * loss_kd

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                step += 1
                continue

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), tc["grad_clip"])
            if not torch.isfinite(grad_norm):
                # NaN/inf gradients: skip update to protect weights and EMA
                optimizer.zero_grad()
                step += 1
                continue
            optimizer.step()
            scheduler.step()

            if ema is not None:
                ema.update(generator)

            with torch.no_grad():
                preds   = gen_logits.argmax(dim=-1)
                correct = ((preds == token_ids) & ce_mask).sum()
                acc     = (correct / ce_mask_sum.clamp(min=1)).item()

            running_ce    += loss_ce.item()
            running_kd    += loss_kd.item()
            running_acc   += acc

            if (step + 1) % log_every == 0:
                elapsed    = time.time() - t0
                avg_ce     = running_ce    / log_every
                avg_kd     = running_kd    / log_every
                avg_acc    = running_acc   / log_every
                lr         = optimizer.param_groups[0]["lr"]
                print(
                    f"[train_stage_b] step={step+1:6d}  elapsed={elapsed:.0f}s  "
                    f"ce={avg_ce:.4f}  kd={avg_kd:.4f}  acc={avg_acc:.4f}  lr={lr:.2e}"
                )
                wandb.log({
                    "train/loss_ce":    avg_ce,
                    "train/loss_kd":    avg_kd,
                    "train/acc":        avg_acc,
                    "train/lr":         lr,
                    "train/elapsed_s":  elapsed,
                }, step=step + 1)
                running_ce = running_kd = running_acc = 0.0

            if (step + 1) % sample_every == 0:
                eval_gen   = ema.shadow if ema else generator
                if arch == "latent_mdm":
                    sample_ids = generate_mdm_samples(
                        eval_gen, num_samples_gen, device, mask_token_id,
                        z_scale=sample_z_scale, num_steps=mdm_steps,
                        tau=mdm_tau, eta=mdm_eta,
                    )
                else:
                    sample_ids = generate_samples(eval_gen, num_samples_gen, device,
                                                  z_scale=sample_z_scale,
                                                  temperature=0.9, top_p=0.95)
                texts = decode_tokens(sample_ids)
                print(f"\n[samples] step={step+1}")
                for i, t in enumerate(texts):
                    print(f"  [{i}] {t[:200]}")
                print()
                table = wandb.Table(columns=["step", "sample_id", "text"])
                for i, t in enumerate(texts):
                    table.add_data(step + 1, i, t)
                wandb.log({"samples/text": table}, step=step + 1)
                generator.train()

            if eval_sweep_every and (step + 1) % eval_sweep_every == 0:
                eval_gen = ema.shadow if ema else generator
                _run_mdm_eval_sweep(
                    eval_gen, cfg, device, step + 1,
                    mdm_steps_list=eval_mdm_steps,
                    mdm_tau=eval_mdm_tau,
                    mdm_eta=eval_mdm_eta,
                    z_scale=sample_z_scale,
                    num_samples=eval_num_samples,
                    gpt2_model=eval_gpt2_model,
                    batch_size=eval_batch_size,
                )
                generator.train()

            # Periodic mid-run checkpoint every 25k steps so early termination
            # doesn't lose all progress.
            if (step + 1) % 25000 == 0:
                _save_checkpoint(ckpt_path_b, step + 1, generator, optimizer, ema, cfg)
                print(f"[train_stage_b] Checkpoint saved at step {step+1}")

            step += 1

        if step >= max_steps:
            break

    elapsed = time.time() - t0
    print(f"\n[train_stage_b] Finished: {step} steps in {elapsed:.1f}s")

    _save_checkpoint(ckpt_path_b, step, generator, optimizer, ema, cfg)
    print(f"[train_stage_b] Saved {ckpt_path_b}")

    print("\n[eval] Generating final samples...")
    eval_gen = ema.shadow if ema else generator
    eval_gen.eval()

    if arch == "latent_mdm":
        sample_ids = generate_mdm_samples(
            eval_gen, final_num_samples, device, mask_token_id,
            z_scale=sample_z_scale, num_steps=mdm_steps,
            tau=mdm_tau, eta=mdm_eta,
        )
    else:
        sample_ids = generate_samples(eval_gen, final_num_samples, device, z_scale=sample_z_scale)
    texts = decode_tokens(sample_ids)
    print("\n=== Generated text samples (argmax decoding) ===")
    for i, t in enumerate(texts):
        print(f"  [{i:2d}] {t[:300]}")

    total_correct = total_tokens = 0
    total_ce = 0.0
    n_batches = 0
    eval_gen.eval()
    with torch.no_grad():
        for token_ids, padding_mask in val_loader:
            token_ids    = token_ids.to(device)
            padding_mask = padding_mask.to(device)
            u_mean = stage_a.encode(token_ids)
            z, _   = stage_a.flow(u_mean)
            if arch == "latent_mdm":
                xt = token_ids.masked_fill(padding_mask.bool(), mask_token_id)
                logits = eval_gen(xt, z)
            else:
                logits = eval_gen(z)
            preds  = logits.argmax(dim=-1)
            total_correct += ((preds == token_ids) & padding_mask).sum().item()
            total_tokens  += padding_mask.sum().item()
            B, T, V = logits.shape
            ce_all = F.cross_entropy(
                logits.reshape(-1, V), token_ids.reshape(-1), reduction="none"
            ).reshape(B, T)
            total_ce += (ce_all * padding_mask).sum().item()
            n_batches += 1
            if n_batches >= eval_batches:
                break

    gen_acc = total_correct / max(total_tokens, 1)
    gen_ce  = total_ce      / max(total_tokens, 1)
    print(f"\ngen_acc={gen_acc:.4f}")
    print(f"gen_ce={gen_ce:.4f}")
    print(f"steps={step}")
    print(f"elapsed={elapsed:.1f}")

    wandb.log({"eval/gen_acc": gen_acc, "eval/gen_ce": gen_ce}, step=step)

    if wb.get("save_artifact", True):
        artifact = wandb.Artifact("stage_b_checkpoint", type="model")
        artifact.add_file(str(ckpt_path_b))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train()
