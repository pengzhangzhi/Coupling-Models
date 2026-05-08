# transport_model_mnist

Repository for experiments around one-step generative transport models and MNIST guidance comparisons.

## What lives here

- `mnist_guidance/`
  - MNIST guidance experiments and baselines.
  - `coupling_model/` contains the one-step generator + reward guidance workflow.
  - `mdm_baseline/` contains the diffusion baseline used for comparison.
- `one-step-text-lm1b/`
  - LM1B text-generation experiments for the coupling-model paper.
  - Includes data prep, Stage A / Stage B training, and evaluation scripts.
- `one-step-DNA/`
  - DNA-related one-step model code.
- `Coupling_Models_NeurIPS_2026 (1).pdf`
  - Paper PDF kept in the repo root.

## HPC setup

This repo is meant to run on the Isambard AI HPC environment with scratch-first paths for large files.

Source your shell aliases first:

```bash
source ~/.alias
```

Useful commands from that setup:

```bash
uvproj transport_model_mnist
workon transport_model_mnist
```

The scratch layout is defined by the shell config:

- `ISB_SCRATCH_ROOT`
- `ISB_PROJECT_ROOT`
- `ISB_RUN_ROOT`
- `ISB_CACHE_ROOT`

Large outputs such as checkpoints, samples, and W&B runs should live under scratch, not in the source tree.

## MNIST guidance examples

The `mnist_guidance/coupling_model/README.md` file has the current commands for:

- CFG sweeps
- reward-guidance evaluation

The `mnist_guidance/mdm_baseline/README.md` file has the matching MDM baseline commands.

## LM1B text example

The `one-step-text-lm1b/README.md` file documents the LM1B pipeline, including:

- data preparation
- Stage A training
- Stage B training
- evaluation and sweeps

## Notes

- Checkpoints are intentionally not tracked in git.
- The repo root does not contain a unified training entrypoint; use the subproject READMEs for runnable commands.
