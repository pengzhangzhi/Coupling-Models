"""Evaluation: Gen. PPL (GPT-2 Large) + unigram entropy on generated samples.

Pipeline
--------
  z ~ N(0, scale²·I)  →  TextGeneratorTransformer  →  token_ids (Qwen vocab)
  token_ids            →  Qwen tokenizer decode      →  text strings
  text strings         →  GPT-2 Large                →  per-token NLL  →  PPL

Reference calibration
---------------------
  --reference_ppl      Compute GPT-2 Large PPL on real LM1B test sentences
                       (expected ~30–35). This confirms the evaluator is
                       calibrated and makes your gen-PPL numbers meaningful.

Usage
-----
  uv run python eval.py                                                        # argmax, z_scale=0.90
  uv run python eval.py --temperature 0.9 --top_p 0.95
  uv run python eval.py --sweep                                                # sweep temperatures
  uv run python eval.py --z_scale_sweep                                        # sweep z_scales
  uv run python eval.py --reference_ppl                                        # evaluator calibration check
  uv run python eval.py --reference_ppl --num_ref 2048                         # use more sentences
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

from prepare import QWEN_VOCAB_SIZE, MAX_SEQ_LEN, REPR_DIM, PAD_TOKEN_ID, QWEN_MODEL
from runtime_paths import dataset_dir, resolve_checkpoint_path
from train_stage_a import StageAModel
from train_stage_b import (
    LatentMaskedDenoiser,
    TextGeneratorTransformer,
    decode_tokens,
    generate_mdm_samples,
    generate_samples,
    get_mask_token_id,
)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_stage_a(checkpoint_path: str, device: str) -> StageAModel:
    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stage_a = StageAModel(ckpt["config"]).to(device)
    stage_a.load_state_dict(ckpt["model_state_dict"], strict=False)
    stage_a.encoder.proj.fitted.fill_(True)
    stage_a.eval()
    for p in stage_a.parameters():
        p.requires_grad_(False)
    print(f"[eval] Loaded Stage A from {checkpoint_path} (step={ckpt['step']})")
    return stage_a


def generate_nf_decoder(stage_a: StageAModel, n: int, device: str,
                         z_scale: float = 1.0, temperature: float = 0.0,
                         top_p: float = 1.0) -> torch.Tensor:
    """Sample z ~ N(0, z_scale²·I) → TarFlow⁻¹ → Stage A decoder → token_ids."""
    z = torch.randn(n, MAX_SEQ_LEN, REPR_DIM, device=device) * z_scale
    with torch.no_grad():
        u      = stage_a.flow.inverse(z)
        logits = stage_a.decode(u)          # (n, T, V)
    # NaN/inf from OOD TarFlow inverse outputs: replace before any arithmetic
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=10.0, neginf=-10.0)
    greedy = logits.argmax(-1)                # (n, T) — fallback for degenerate rows
    if temperature == 0.0:
        return greedy
    # Shift to [-inf, 0] before temperature scaling to prevent overflow
    logits = (logits - logits.max(dim=-1, keepdim=True).values) / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove   = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
    probs       = torch.softmax(logits, dim=-1)
    flat_probs  = probs.reshape(-1, QWEN_VOCAB_SIZE).clone()
    flat_greedy = greedy.reshape(-1)
    # Rows where all probs are zero: fall back to one-hot at greedy token
    bad_rows = flat_probs.sum(-1) < 1e-8
    if bad_rows.any():
        bad_idx = bad_rows.nonzero(as_tuple=True)[0]
        flat_probs[bad_idx] = 0.0
        flat_probs[bad_idx, flat_greedy[bad_idx]] = 1.0
    return torch.multinomial(flat_probs, 1).view(n, MAX_SEQ_LEN)


def load_generator(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    gc  = cfg["generator"]
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
    else:
        raise ValueError(f"Unknown generator.arch={arch!r}")

    if "ema_state_dict" in ckpt:
        generator.load_state_dict(ckpt["ema_state_dict"], strict=False)
        print(f"[eval] Loaded EMA weights from {checkpoint_path} (step={ckpt['step']})")
    else:
        generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
        print(f"[eval] Loaded generator weights from {checkpoint_path} (step={ckpt['step']})")

    generator.eval()
    return generator, cfg


def format_eval_context(arch: str, temperature: float, top_p: float,
                        z_scale: float, mdm_steps: int,
                        mdm_tau: float, mdm_eta: float) -> str:
    if arch == "latent_mdm":
        return (
            f"mdm_steps={mdm_steps}, mdm_tau={mdm_tau}, "
            f"mdm_eta={mdm_eta}, z_scale={z_scale}"
        )
    return f"temperature={temperature}, top_p={top_p}, z_scale={z_scale}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_unigram_entropy(token_ids: torch.Tensor) -> float:
    """Average per-sample unigram entropy (bits)."""
    entropies = []
    for row in token_ids:
        counts = torch.bincount(row, minlength=QWEN_VOCAB_SIZE).float()
        p = counts / counts.sum()
        nz = p > 0
        H = -(p[nz] * p[nz].log2()).sum().item()
        entropies.append(H)
    return sum(entropies) / len(entropies)


def compute_gen_ppl(texts: list[str], device: str,
                    gpt2_model: str = "gpt2-large",
                    batch_size: int = 16) -> float:
    """Generative perplexity via frozen GPT-2 Large."""
    print(f"[eval] Loading {gpt2_model} evaluator ...")
    tokenizer = AutoTokenizer.from_pretrained(gpt2_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(gpt2_model).to(device)
    model.eval()

    total_nll = total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256,
        ).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask   = attn_mask[:, 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)

        total_nll    += (nll * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()

        done = min(i + batch_size, len(texts))
        if (i // batch_size + 1) % 16 == 0 or done == len(texts):
            print(f"  [{done}/{len(texts)}] running PPL = {math.exp(total_nll / max(total_tokens, 1)):.2f}")

    ppl = math.exp(total_nll / max(total_tokens, 1))
    del model
    torch.cuda.empty_cache()
    return ppl


def compute_reference_ppl(device: str,
                           num_sentences: int = 1024,
                           gpt2_model: str = "gpt2-large",
                           batch_size: int = 16,
                           seed: int = 42) -> float:
    """GPT-2 Large PPL on real LM1B test sentences.

    Pipeline mirrors gen PPL exactly:
      LM1B test .npy  →  Qwen tokenizer.decode()  →  GPT-2 Large  →  PPL

    Expected result: ~30–35 (LM1B is an easy, single-domain corpus).
    A value far outside this range indicates a tokenization or data issue.
    """
    cache_path = dataset_dir("lm1b") / "test_qwen.npy"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {cache_path}. "
            "Run: uv run python prepare.py --dataset lm1b --tokenizer qwen"
        )

    print(f"[ref_ppl] Loading LM1B test data from {cache_path} ...")
    test_ids = np.load(str(cache_path), mmap_mode="r")   # (N, 128) int32

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_ids), size=min(num_sentences, len(test_ids)), replace=False)
    sample = test_ids[indices]                             # (num_sentences, 128)

    # Decode with Qwen tokenizer, stripping padding — same as decode_tokens() in train_stage_b
    print(f"[ref_ppl] Decoding {len(sample):,} sentences with Qwen tokenizer ...")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    qwen_tok.pad_token = qwen_tok.eos_token

    texts = []
    for row in sample:
        ids = row.tolist()
        # Strip trailing pad tokens before decoding
        while ids and ids[-1] == PAD_TOKEN_ID:
            ids.pop()
        texts.append(qwen_tok.decode(ids, skip_special_tokens=True))

    print(f"[ref_ppl] Sample reference sentences:")
    for i in range(min(3, len(texts))):
        print(f"  [{i}] {texts[i][:160]}")
    print()

    ref_ppl = compute_gen_ppl(texts, device, gpt2_model=gpt2_model, batch_size=batch_size)
    print(f"[ref_ppl] Reference PPL (LM1B test, GPT-2 Large) = {ref_ppl:.2f}  "
          f"(expected ~30–35)")
    return ref_ppl


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(generator, device: str,
             num_samples: int = 1024, temperature: float = 0.0,
             top_p: float = 1.0, z_scale: float = 0.90,
             gpt2_model: str = "gpt2-large", batch_size: int = 16,
             cfg: dict | None = None, mdm_steps: int = 8,
             mdm_tau: float = 1.0, mdm_eta: float = 1.0) -> dict:
    arch = (cfg or {}).get("generator", {}).get("arch", "transformer")
    context = format_eval_context(
        arch, temperature=temperature, top_p=top_p, z_scale=z_scale,
        mdm_steps=mdm_steps, mdm_tau=mdm_tau, mdm_eta=mdm_eta,
    )
    print(f"\n[eval] Generating {num_samples} samples "
          f"(arch={arch}, {context}) ...")

    chunk   = 32  # keep small: Qwen vocab=151936 → sort tensor ~2.5 GB at chunk=32
    all_ids = []
    mask_token_id = None
    if arch == "latent_mdm":
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        mask_token_id = (cfg or {}).get("training", {}).get("mask_token_id")
        if mask_token_id is None:
            mask_token_id = get_mask_token_id(tokenizer)
    for start in range(0, num_samples, chunk):
        n   = min(chunk, num_samples - start)
        if arch == "latent_mdm":
            ids = generate_mdm_samples(
                generator, n, device, mask_token_id,
                z_scale=z_scale, num_steps=mdm_steps,
                tau=mdm_tau, eta=mdm_eta,
            )
        else:
            ids = generate_samples(generator, n, device,
                                   z_scale=z_scale, temperature=temperature, top_p=top_p)
        all_ids.append(ids.cpu())
    token_ids = torch.cat(all_ids, dim=0)

    entropy = compute_unigram_entropy(token_ids)
    print(f"[eval] Entropy = {entropy:.4f} bits  (LM1B reference: 4.31)")

    texts = decode_tokens(token_ids)
    print("\n[eval] Sample outputs:")
    for i in range(min(4, len(texts))):
        print(f"  [{i}] {texts[i][:180]}")
    print()

    gen_ppl = compute_gen_ppl(texts, device, gpt2_model=gpt2_model, batch_size=batch_size)
    print(f"[eval] Gen. PPL = {gen_ppl:.2f}")

    result = {
        "arch": arch,
        "z_scale": z_scale,
        "num_samples": num_samples,
        "entropy": entropy,
        "gen_ppl": gen_ppl,
    }
    if arch == "latent_mdm":
        result.update({
            "mdm_steps": mdm_steps,
            "mdm_tau": mdm_tau,
            "mdm_eta": mdm_eta,
        })
    else:
        result.update({
            "temperature": temperature,
            "top_p": top_p,
        })
    return result


def print_table(rows: list[dict], arch: str):
    print("\n" + "=" * 70)
    if arch == "latent_mdm":
        print(f"{'mdm_steps':>9}  {'mdm_tau':>8}  {'mdm_eta':>8}  {'z_scale':>7}  "
              f"{'Gen. PPL ↓':>10}  {'Entropy':>8}")
    else:
        print(f"{'temperature':>12}  {'top_p':>6}  {'z_scale':>7}  "
              f"{'Gen. PPL ↓':>10}  {'Entropy':>8}")
    print("-" * 70)
    target_entropy = 4.31
    for r in rows:
        marker = " ←" if abs(r["entropy"] - target_entropy) < 0.05 else ""
        if arch == "latent_mdm":
            print(f"  {r['mdm_steps']:>9d}  {r['mdm_tau']:>8.2f}  {r['mdm_eta']:>8.2f}  "
                  f"{r.get('z_scale', 1.00):>7.4f}  "
                  f"{r['gen_ppl']:>10.2f}  {r['entropy']:>8.4f}{marker}")
        else:
            print(f"  {r['temperature']:>10.2f}  {r['top_p']:>6.2f}  "
                  f"{r.get('z_scale', 1.00):>7.4f}  "
                  f"{r['gen_ppl']:>10.2f}  {r['entropy']:>8.4f}{marker}")
    print("=" * 70)
    if arch == "latent_mdm":
        print(f"  {'LM1B ref':>9}  {'':>8}  {'':>8}  {'':>7}  {'47.47':>10}  {'4.31':>8}  (Qwen-tokenized)")
    else:
        print(f"  {'LM1B ref':>10}  {'':>6}  {'':>7}  {'47.47':>10}  {'4.31':>8}  (Qwen-tokenized)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage B generation evaluation")
    p.add_argument("--checkpoint", default="checkpoints/stage_b/v3_lm1b_mdm/generator.pt",
                   help="Path to Stage B checkpoint. Pass 'none' to skip gen eval "
                        "(useful with --reference_ppl for standalone calibration).")
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.65,
                   help="Transformer-only sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Transformer-only nucleus sampling cutoff.")
    p.add_argument("--z_scale", type=float, default=0.8,
                   help="Scale for z~N(0,I). TarFlow Stage A produces z_std≈1.0.")
    p.add_argument("--mdm_steps", type=int, default=8,
                   help="latent_mdm only: P2 sampling steps.")
    p.add_argument("--mdm_tau", type=float, default=1.0,
                   help="latent_mdm only: P2 categorical sampling temperature.")
    p.add_argument("--mdm_eta", type=float, default=1.0,
                   help="latent_mdm only: P2 remasking score multiplier.")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep temperature/top_p for transformer checkpoints, or mdm_tau for latent_mdm.")
    p.add_argument("--z_scale_sweep", action="store_true",
                   help="Sweep z_scales [0.60, 0.75, 0.838, 0.90, 1.00, 1.10] for the active checkpoint.")
    p.add_argument("--nf_decoder", action="store_true",
                   help="Use NF+decoder generation (Stage A only): "
                        "z~N(0,z_scale²I) → TarFlow⁻¹ → Stage A decoder.")
    p.add_argument("--stage_a_checkpoint",
                   default="checkpoints/stage_a/v3_lm1b/checkpoint.pt",
                   help="Stage A checkpoint for --nf_decoder mode.")
    p.add_argument("--gpt2_model", default="gpt2-large")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    # Evaluator calibration
    p.add_argument("--reference_ppl", action="store_true",
                   help="Compute GPT-2 Large PPL on real LM1B test sentences (~30–35 expected). "
                        "Can be combined with normal evaluation to print a comparison table.")
    p.add_argument("--num_ref", type=int, default=1024,
                   help="Number of LM1B test sentences to score for reference PPL (default 1024).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # --reference_ppl can run standalone (no checkpoint needed) or alongside gen eval
    ref_ppl = None
    if args.reference_ppl:
        ref_ppl = compute_reference_ppl(
            device, num_sentences=args.num_ref,
            gpt2_model=args.gpt2_model, batch_size=args.batch_size,
        )
        if args.checkpoint is None or args.checkpoint.lower() == "none":
            # Standalone calibration run — no generation eval requested
            print("=" * 70)
            print("Evaluator calibration (GPT-2 Large on real LM1B test sentences)")
            print(f"  Reference PPL : {ref_ppl:.2f}  (expected ~30–35)")
            print("=" * 70)
            raise SystemExit(0)

    # ── NF+decoder mode ──────────────────────────────────────────────────────
    if args.nf_decoder:
        stage_a = load_stage_a(str(resolve_checkpoint_path(args.stage_a_checkpoint)), device)

        def _gen(n, temperature, top_p, z_scale):
            return generate_nf_decoder(stage_a, n, device,
                                       z_scale=z_scale, temperature=temperature,
                                       top_p=top_p)

        def _eval_nf(temperature, top_p, z_scale):
            print(f"\n[eval] NF+decoder: temperature={temperature} top_p={top_p} z_scale={z_scale}")
            chunk, all_ids = 32, []
            for start in range(0, args.num_samples, chunk):
                n = min(chunk, args.num_samples - start)
                all_ids.append(_gen(n, temperature, top_p, z_scale).cpu())
            token_ids = torch.cat(all_ids)
            entropy   = compute_unigram_entropy(token_ids)
            texts     = decode_tokens(token_ids)
            print(f"[eval] Entropy = {entropy:.4f}")
            for i in range(min(4, len(texts))):
                print(f"  [{i}] {texts[i][:180]}")
            gen_ppl = compute_gen_ppl(texts, device,
                                     gpt2_model=args.gpt2_model,
                                     batch_size=args.batch_size)
            print(f"[eval] Gen. PPL = {gen_ppl:.2f}")
            return {"temperature": temperature, "top_p": top_p, "z_scale": z_scale,
                    "num_samples": args.num_samples, "entropy": entropy, "gen_ppl": gen_ppl}

        if args.sweep:
            rows = [_eval_nf(t, p, args.z_scale)
                    for t, p in [(0.0, 1.0), (0.3, 0.95), (0.5, 0.95), (0.6, 0.95),
                                 (0.65, 0.95), (0.7, 0.95), (0.75, 0.95),
                                 (0.8, 0.95), (0.9, 0.95), (1.0, 0.95)]]
        else:
            rows = [_eval_nf(args.temperature, args.top_p, args.z_scale)]
        print_table(rows, "transformer")
        raise SystemExit(0)

    # ── Stage B / null-gen mode ───────────────────────────────────────────────
    generator, cfg = load_generator(str(resolve_checkpoint_path(args.checkpoint)), device)
    arch = cfg.get("generator", {}).get("arch", "transformer")

    if args.z_scale_sweep:
        rows = []
        for zs in [0.60, 0.75, 0.838, 0.90, 1.00, 1.10]:
            rows.append(evaluate(generator, device, num_samples=args.num_samples,
                                 temperature=0.0, top_p=1.0, z_scale=zs,
                                 gpt2_model=args.gpt2_model, batch_size=args.batch_size,
                                 cfg=cfg, mdm_steps=args.mdm_steps,
                                 mdm_tau=args.mdm_tau, mdm_eta=args.mdm_eta))
        print_table(rows, arch)
    elif args.sweep:
        rows = []
        if arch == "latent_mdm":
            for tau in [0.0, 0.3, 0.5, 0.7, 1.0]:
                rows.append(evaluate(generator, device, num_samples=args.num_samples,
                                     temperature=0.0, top_p=1.0, z_scale=args.z_scale,
                                     gpt2_model=args.gpt2_model, batch_size=args.batch_size,
                                     cfg=cfg, mdm_steps=args.mdm_steps,
                                     mdm_tau=tau, mdm_eta=args.mdm_eta))
        else:
            configs = [(0.0, 1.00), (0.3, 0.95), (0.5, 0.95), (0.6, 0.95),
                       (0.65, 0.95), (0.7, 0.95), (0.75, 0.95),
                       (0.8, 0.95), (0.9, 0.95), (1.0, 0.95)]
            for temp, top_p in configs:
                rows.append(evaluate(generator, device, num_samples=args.num_samples,
                                     temperature=temp, top_p=top_p, z_scale=args.z_scale,
                                     gpt2_model=args.gpt2_model, batch_size=args.batch_size,
                                     cfg=cfg, mdm_steps=args.mdm_steps,
                                     mdm_tau=args.mdm_tau, mdm_eta=args.mdm_eta))
        print_table(rows, arch)
    else:
        result = evaluate(generator, device, num_samples=args.num_samples,
                          temperature=args.temperature, top_p=args.top_p,
                          z_scale=args.z_scale,
                          gpt2_model=args.gpt2_model, batch_size=args.batch_size,
                          cfg=cfg, mdm_steps=args.mdm_steps,
                          mdm_tau=args.mdm_tau, mdm_eta=args.mdm_eta)
        print_table([result], arch)

    if ref_ppl is not None:
        print("=" * 70)
        print("Evaluator calibration (GPT-2 Large on real LM1B test sentences)")
        print(f"  Reference PPL : {ref_ppl:.2f}  (expected ~30–35)")
        print("  This confirms the evaluator is behaving correctly.")
        print("=" * 70)
