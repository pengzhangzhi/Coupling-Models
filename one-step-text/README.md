# one-step-text

Local-first text generation experiments for the transport model release.

## Layout

- `prepare.py` downloads and tokenizes datasets.
- `train_stage_a.py` trains Stage A.
- `train_stage_b.py` trains Stage B.
- `eval.py` runs generation and perplexity evaluation.
- `configs/` contains the YAML configs for the released runs.

## Setup

Install dependencies from the repo root:

```bash
uv sync
```

If you want to place generated data, checkpoints, temporary files, and W&B
logs somewhere else, set `LTLM_RUNTIME_ROOT` before running commands. By
default they are stored under `~/.local/share/latent-transport-lm/`.

## Data Prep

Prepare the LM1B Qwen-tokenized data:

```bash
uv run python prepare.py --dataset lm1b --tokenizer qwen
```

Prepare the OpenWebText Qwen-tokenized data:

```bash
uv run python prepare.py --dataset owt --tokenizer qwen
```

If you need the BERT-tokenized variants for the MDLM baseline:

```bash
uv run python prepare.py --dataset lm1b --tokenizer bert
uv run python prepare.py --dataset owt --tokenizer bert
```

## Training

Stage A:

```bash
uv run python train_stage_a.py --dataset lm1b --config configs/stage_a.yaml
```

Stage B transformer:

```bash
uv run python train_stage_b.py --dataset lm1b --config configs/stage_b.yaml
```

Stage B latent MDM:

```bash
uv run python train_stage_b.py --dataset lm1b --config configs/stage_b_mdm.yaml
```

These commands write checkpoints under the runtime checkpoint root, for
example:

```text
~/.local/share/latent-transport-lm/checkpoints/...
```

## Eval

Evaluate a checkpoint:

```bash
uv run python eval.py --checkpoint checkpoints/stage_b/v3_lm1b_mdm/generator.pt
```

Run the LM1B GPT-2 Large calibration check:

```bash
uv run python eval.py --reference_ppl
```

For NF-decoder evaluation, pass a Stage A checkpoint explicitly:

```bash
uv run python eval.py \
  --nf_decoder \
  --stage_a_checkpoint checkpoints/stage_a/v3_lm1b/checkpoint.pt
```

## Notes

- The release does not ship shell helpers or Slurm wrappers.
- W&B settings come from your environment or the default config fields.
- File paths in the configs are relative to the runtime checkpoint root.
