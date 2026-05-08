# Coupling Models: One-Step LM1B Text Generation

Official implementation for the LM1B text experiment in **Coupling Models for One-Step Discrete Generation**.

This code trains a two-stage latent coupling model for unconditional language generation. Stage A maps Qwen-tokenized LM1B sequences into Gaussian latents with a frozen Qwen2.5 encoder and a TarFlow normalizing flow. Stage B trains a parallel Transformer decoder to invert those latents into full token sequences in one forward pass.

## Method

```
tokens x
  -> frozen Qwen2.5 encoder
  -> hidden states h
  -> TarFlow f_phi(h) = z
  -> one-step Transformer G_theta(z)
  -> token logits
```

At inference time, Stage A is not used. Sampling is simply:

```python
z = torch.randn(batch, seq_len, latent_dim)
logits = generator(z)
tokens = sample(logits, temperature=tau, top_p=0.95)
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `prepare.py` | Tokenizes LM1B/OpenWebText into memory-mapped arrays. |
| `train_stage_a.py` | Trains the frozen-Qwen + TarFlow latent coupling. |
| `train_stage_b.py` | Trains the one-step Transformer generator with Qwen KD. |
| `eval.py` | Computes GPT-2-Large generative perplexity and unigram entropy. |
| `configs/` | Public training configs for Stage A and Stage B. |
| `runtime_paths.py` | Centralized data/checkpoint/runtime path resolution. |
| `scripts/` | Generic setup and pipeline helpers. |

## Installation

The project uses `uv`. Runtime artifacts are written outside the source tree when `LTLM_RUNTIME_ROOT` or `SCRATCH` is set; otherwise they go under `.runtime/`.

```bash
cd one-step-text-lm1b

# Optional but recommended for large runs:
export LTLM_RUNTIME_ROOT=/path/to/runtime/latent-transport-lm

bash scripts/setup_env.sh --create-venv
source scripts/setup_env.sh
```

The setup script creates:

```text
$LTLM_RUNTIME_ROOT/
  data/
  checkpoints/
  wandb/
  tmp/
  .venv/
```

## Data

Prepare LM1B with the Qwen2.5 tokenizer:

```bash
scripts/run.sh python prepare.py --dataset lm1b --tokenizer qwen
```

This writes:

```text
$LTLM_DATA_ROOT/lm1b/train_qwen.npy
$LTLM_DATA_ROOT/lm1b/test_qwen.npy
```

## Training

Use the same `EXPERIMENT_NAME` for both stages so Stage B can find the Stage A checkpoint.

```bash
export EXPERIMENT_NAME=coupling_lm1b

scripts/run.sh python train_stage_a.py \
  --dataset lm1b \
  --config configs/stage_a.yaml

scripts/run.sh python train_stage_b.py \
  --dataset lm1b \
  --config configs/stage_b.yaml
```

The convenience pipeline runs data preparation followed by both stages:

```bash
EXPERIMENT_NAME=coupling_lm1b scripts/run_pipeline.sh
```

For offline logging:

```bash
WANDB_MODE=offline scripts/run_pipeline.sh
```

## Evaluation

Evaluate a trained Stage B checkpoint:

```bash
scripts/run.sh python eval.py \
  --checkpoint stage_b/v3_lm1b/coupling_lm1b/checkpoint.ckpt \
  --temperature 0.65 \
  --top_p 0.95 \
  --z_scale 0.8
```

Temperature and latent-scale sweeps:

```bash
scripts/run.sh python eval.py \
  --checkpoint stage_b/v3_lm1b/coupling_lm1b/checkpoint.ckpt \
  --sweep

scripts/run.sh python eval.py \
  --checkpoint stage_b/v3_lm1b/coupling_lm1b/checkpoint.ckpt \
  --z_scale_sweep
```

Calibrate the evaluator on real LM1B test sentences:

```bash
scripts/run.sh python eval.py --checkpoint none --reference_ppl --num_ref 1024
```

## Notes

- The released text implementation uses full-dimensional Qwen hidden states (`d=896`) with TarFlow.
- Stage B uses token cross-entropy plus KL distillation from a frozen Qwen2.5-0.5B causal LM.
- The default configs assume multi-GPU training. For a single GPU, set `lightning.devices: 1` and remove `lightning.strategy: ddp` in the YAML files.
- Checkpoints are intentionally not stored in the repository.

## Citation

```bibtex
@inproceedings{peng2026couplingmodels,
  title     = {Coupling Models for One-Step Discrete Generation},
  author    = {Peng, Fred Zhangzhi and Bose, Avishek Joey and Zhang, Anru R. and Tong, Alexander},
  year      = {2026}
}
```
