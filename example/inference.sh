cd /home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-

export CUDA_VISIBLE_DEVICES=4,5

python3 abstract_metrics_from_pretrained_ddpm.py \
    checkpoint_path="/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-/Checkpoints/ep15_bs40_T1000_lr1e-4/ckpt_0_.pt" \
    use_multi_gpu=true \
    device=cuda \
    device_ids="[0,1]" \




















