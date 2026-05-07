# 32x32 U-Net MDM Baseline

For the guidance comparison in the paper, use only the 32x32 U-Net MDM
checkpoint:

```bash
"$ISB_RUN_ROOT/unet_mdm_baseline/checkpoints/best.pt"
```

The older direct-pixel MDM checkpoint at
`"$ISB_RUN_ROOT/mdm_baseline/checkpoints/last.pt"` is not part of the current
guidance study.

Token convention:

- `0`: black pixel
- `1`: white pixel
- `2`: `[MASK]`

## Train U-Net MDM

```bash
uvproj transport_model_mnist
workon transport_model_mnist

CUDA_VISIBLE_DEVICES=0 python -m mdm_baseline.train_unet_mdm \
  --data-dir ./data \
  --output-dir "$ISB_RUN_ROOT/unet_mdm_baseline" \
  --epochs 100 \
  --batch-size 192 \
  --eval-batch-size 384 \
  --num-workers 8 \
  --objective md4 \
  --base-channels 128 \
  --channel-mult 1,2,2,4 \
  --num-res-blocks 2 \
  --attention-resolutions 8,16 \
  --num-heads 4 \
  --cfg-drop-prob 0.1 \
  --sample-every 5 \
  --sample-steps 256 \
  --sample-sampler md4
```

## Evaluate U-Net MDM + CFG

```bash
uvproj transport_model_mnist
workon transport_model_mnist

CUDA_VISIBLE_DEVICES=0 python -m mdm_baseline.eval_unet_mdm_cfg \
  --checkpoint-path "$ISB_RUN_ROOT/unet_mdm_baseline/checkpoints/best.pt" \
  --eval-classifier-checkpoint outputs/eval_classifier.pt \
  --output-dir "$ISB_RUN_ROOT/guidance_eval_unet_paper_seed0/unet_cfg" \
  --seed 0 \
  --target-class none \
  --num-samples 1000 \
  --batch-size 64 \
  --steps 256 \
  --sampler md4 \
  --cfg-scales 0 0.25 0.5 0.75 1 1.25 1.5 2 3 4 \
  --fid-num-workers 0
```

`--target-class none`, `--num-samples 1000`, `--seed 0`, and the CFG scale grid
match the one-step generator CFG sweep protocol. `cfg_scale=1` uses one
conditional denoiser call per step. `cfg_scale=0` uses one unconditional call per
step. Other CFG scales use conditional and unconditional calls.

## Evaluate U-Net MDM + Classifier Guidance

```bash
uvproj transport_model_mnist
workon transport_model_mnist

CUDA_VISIBLE_DEVICES=0 python -m mdm_baseline.eval_unet_mdm_classifier_guidance \
  --checkpoint-path "$ISB_RUN_ROOT/unet_mdm_baseline/checkpoints/best.pt" \
  --reward-checkpoint outputs/reward_model/checkpoints/last.pt \
  --eval-classifier-checkpoint outputs/eval_classifier.pt \
  --output-dir "$ISB_RUN_ROOT/guidance_eval_unet_paper_seed0/unet_classifier" \
  --target-class none \
  --num-samples 1000 \
  --batch-size 32 \
  --steps 256 \
  --sampler md4 \
  --guidance-scales 0 0.5 1 2 \
  --seed 0 \
  --fid-num-workers 0
```

## U-Net MDM DRAKES-Style Reward Fine-Tuning

```bash
uvproj transport_model_mnist
workon transport_model_mnist

CUDA_VISIBLE_DEVICES=0 python -m mdm_baseline.finetune_unet_mdm_drakes \
  --checkpoint-path "$ISB_RUN_ROOT/unet_mdm_baseline/checkpoints/best.pt" \
  --reward-checkpoint outputs/reward_model/checkpoints/last.pt \
  --eval-classifier-checkpoint outputs/eval_classifier.pt \
  --output-dir "$ISB_RUN_ROOT/guidance_eval_unet_paper_seed0/unet_drakes_t200" \
  --target-class none \
  --steps 200 \
  --batch-size 4 \
  --sample-steps 16 \
  --eval-sample-steps 256 \
  --eval-num-samples 1000 \
  --eval-batch-size 64 \
  --save-num-samples 100 \
  --seed 0 \
  --fid-num-workers 0
```

The fine-tuning script uses a straight-through binary Concrete relaxed reverse
trajectory and a per-step KL anchor to the frozen pretrained U-Net MDM. The
training trajectory can use fewer relaxed reverse steps than the final
evaluation trajectory to keep reward fine-tuning feasible for the large U-Net.
