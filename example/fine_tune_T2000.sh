#!/bin/bash
# Fine-tuning script for extending T from 1000 to 2000

cd /home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-

# Set GPU devices (will be remapped to 0,1)
export CUDA_VISIBLE_DEVICES=4,5

# Run fine-tuning
python3 fine_tune_extended_T.py \
    checkpoint_path="/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-/Checkpoints/ep15_bs40_T1000_lr1e-4/ckpt_0_.pt" \
    T=2000 \
    fine_tune_epochs=5 \
    fine_tune_lr=1e-5 \
    batch_size=64 \
    device=cuda \
    use_multi_gpu=true \
    device_ids="[0,1]" \
    save_weight_dir="./Checkpoints/fine_tuned_T2000" \
    imagenet_root="/data0/datasets/imagenet/imagenet"

echo ""
echo "Fine-tuning completed!"
echo "Use the checkpoint in ./Checkpoints/fine_tuned_T2000/ for inference"

