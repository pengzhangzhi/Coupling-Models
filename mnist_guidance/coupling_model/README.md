class-cond one-step generator: /lus/lfs1aip2/scratch/u6dq/fredpeng.u6dq/runs/one_step_mnist_cond/checkpoints/last.pt

reward model: outputs/reward_model/checkpoints/last.pt

eval classifier: outputs/eval_classifier.pt




CFG eval:
```
python eval_cfg_sweep.py   --checkpoint-path /scratch/u6dq/fredpeng.u6dq/runs/one_step_mnist_cond/checkpoints/last.pt   --output-dir outputs/cfg_sweep   --fid-num-gen 1000   --seed 0 
```

reward guidance eval: 
```
uvproj transport_model_mnist
workon transport_model_mnist
 python eval_reward_guidance.py --generator-checkpoint /lus/lfs1aip2/scratch/u6dq/fredpeng.u6dq/runs/one_step_mnist_cond/checkpoints/last.pt  --reward-checkpoint outputs/reward_model/checkpoints/last.pt  --eval-classifier-checkpoint  outputs/eval_classifier.pt
```