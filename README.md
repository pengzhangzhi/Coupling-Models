# Coupling Models

Official Implementation for for experiments around *Coupling Models for One-Step Discrete Generation*.

## What lives here

- `mnist_guidance/`
  - MNIST guidance experiments and baselines.
  - `coupling_model/` contains the one-step generator + reward guidance workflow.
  - `mdm_baseline/` contains the diffusion baseline used for comparison.
- `one-step-text/`
  - LM1B text-generation experiments for the coupling-model paper.
  - Includes data prep, Stage A / Stage B training, and evaluation scripts.
- `one-step-DNA/`
  - DNA-related one-step model code.
