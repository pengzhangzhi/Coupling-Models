# Coupling Models

Official PyTorch Implementation for experiments around [*Coupling Models for One-Step Discrete Generation*](https://arxiv.org/pdf/2605.07193).

## What lives here

- `mnist_guidance/`
  - MNIST guidance experiments and baselines.
  - `coupling_model/` contains the one-step generator + reward guidance workflow.
  - `mdm_baseline/` contains the diffusion baseline used for comparison.
- `one-step-text/`
  - Local-first text generation experiments for the transport model release.
  - Includes data prep, Stage A / Stage B training, and evaluation scripts.
- `one-step-text-lm1b/`
  - Original LM1B text-generation implementation for the coupling-model paper.
  - Includes the paper-oriented training and evaluation scripts.
- `one-step-DNA/`
  - DNA-related one-step model code.
