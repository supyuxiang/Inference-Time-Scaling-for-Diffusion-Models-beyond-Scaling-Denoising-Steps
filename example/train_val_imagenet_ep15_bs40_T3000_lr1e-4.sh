#!/bin/bash
# Training script for DDPM with Hydra configuration management
# Usage: bash train_val_imagenet_ep15_bs40_T3000_lr1e-4.sh
# 
# This script uses Hydra to manage configuration. You can override any parameter:
# - Direct format: python3 Main.py epoch=100 batch_size=512
# - Nested format: python3 Main.py model_config.epoch=100 (legacy, still supported)
#
# Key parameters:
#   T: Training T - number of diffusion steps for training (default: 3000)
#   inference_T: Inference T - number of diffusion steps for sampling/evaluation (default: same as T)
#                If inference_T is different from T, the model trained with T can be used
#                for inference with inference_T steps (thanks to functional time embedding)

cd /home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-

export CUDA_VISIBLE_DEVICES=6,7,8,9

python3 Main.py \
    state=train \
    epoch=15 \
    batch_size=40 \
    T=1000 \
    inference_T=2000 \
    lr=1e-4 \
    use_multi_gpu=true \
    device=cuda \
    save_weight_dir="./Checkpoints/train0.01_val_imagenet_ep15_bs40_T1000-2000_lr1e-4_ratio0.002" \
    metrics_save_dir="./metrics_curves/train0.01_val_imagenet_ep15_bs40_T1000-2000_lr1e-4_ratio0.002" \
    sampled_dir="./SampledImgs/train0.01_val_imagenet_ep15_bs40_T1000-2000_lr1e-4_ratio0.002" \
    training_load_weight=null \
    num_res_blocks=2 \
    dropout=0.15 \
    multiplier=2.0 \
    beta_1=1e-4 \
    beta_T=0.02 \
    img_size=256 \
    grad_clip=1.0 \
    eval_freq=1 \
    eval_metric_interval=20 \
    fid_num_real_samples=3000 \
    clip_num_real_samples=3000 \
    eval_batch_size=16 \
    model_save_freq=1 \
    use_full_dataset=false \
    train_subset_ratio=0.002








