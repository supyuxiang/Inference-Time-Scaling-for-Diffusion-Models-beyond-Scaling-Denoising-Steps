#!/bin/bash
# Training script for DDPM with Hydra configuration management
# Usage: bash ep50_bs1024_T1000_lr1e-4.sh
# 
# This script uses Hydra to manage configuration. You can override any parameter:
# - Direct format: python3 Main.py epoch=100 batch_size=512
# - Nested format: python3 Main.py model_config.epoch=100 (legacy, still supported)

cd /home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-

export CUDA_VISIBLE_DEVICES=6,7,8,9

python3 Main.py \
    state=train \
    epoch=50 \
    batch_size=1024 \
    T=1000 \
    lr=1e-4 \
    use_multi_gpu=true \
    device=cuda \
    save_weight_dir="./Checkpoints/ep50_bs1024_T1000_lr1e-4" \
    metrics_save_dir="./metrics_curves/ep50_bs1024_T1000_lr1e-4" \
    sampled_dir="./SampledImgs/ep50_bs1024_T1000_lr1e-4" \
    training_load_weight=null \
    num_res_blocks=2 \
    dropout=0.15 \
    multiplier=2.0 \
    beta_1=1e-4 \
    beta_T=0.02 \
    img_size=256 \
    grad_clip=1.0








