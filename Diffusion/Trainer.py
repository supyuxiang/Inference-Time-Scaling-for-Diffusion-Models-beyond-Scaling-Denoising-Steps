
import os
from typing import Dict, List, Tuple, Optional
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import sys

sys.path.insert(0, '/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-')
from utils.metrics import FID, IS, CLIPScore

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


class Trainer:
    def __init__(self,cfg:Dict):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.model = UNet(T=cfg["T"], ch=cfg["channel"], ch_mult=cfg["channel_mult"], attn=cfg["attn"],
                     num_res_blocks=cfg["num_res_blocks"], dropout=0.)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=cfg["multiplier"], total_epoch=cfg["epoch"], after_scheduler=None)
        self.fid_calculator = FID()
        self.is_calculator = IS()
        self.clip_calculator = CLIPScore()
        self.real_features = None
        self.real_clip_features = None
        self.mu_real = None
        self.sigma_real = None
        self.metric_interval = cfg.get("metric_interval", 5)
        self.eval_freq = cfg.get("eval_freq", 5)
        self.eval_metric_interval = cfg.get("eval_metric_interval", cfg.get("metric_interval", 20))
        self.fid_num_real_samples = cfg.get("fid_num_real_samples", 5000)
        self.clip_num_real_samples = cfg.get("clip_num_real_samples", 5000)
        self.eval_batch_size = cfg.get("eval_batch_size")
        self.img_size = cfg.get("img_size", 256)
        self.sampled_dir = cfg.get("sampled_dir", "./SampledImgs")
        self.nrow = cfg.get("nrow", 8)
        self.metrics_save_dir = cfg.get("metrics_save_dir", "./metrics_curves")
        self.save_weight_dir = cfg.get("save_weight_dir", "./Checkpoints")
        self.test_load_weight = cfg.get("test_load_weight", "ckpt_199_.pt")
        self.sampledNoisyImgName = cfg.get("sampledNoisyImgName", "NoisyNoGuidenceImgs.png")
        self.sampledImgName = cfg.get("sampledImgName", "SampledNoGuidenceImgs.png")
        self.sampled_dir = cfg.get("sampled_dir", "./SampledImgs")
    
    def sample_with_metrics_tracking(self):
        pass

    def compute_real_features_for_fid(self):
        pass

    def compute_real_features_for_clip(self):
        pass

    def plot_loss_curve(self):
        pass

    def plot_metrics_curves(self):
        pass
    
    def train(self):
        pass


def sample_with_metrics_tracking(
    sampler: GaussianDiffusionSampler,
    x_T: torch.Tensor,
    fid_calculator: FID,
    is_calculator: IS,
    clip_calculator: Optional[CLIPScore],
    real_features: torch.Tensor,
    real_clip_features: torch.Tensor,
    mu_real: torch.Tensor = None,
    sigma_real: torch.Tensor = None,
    metric_interval: int = 5,
    device: str = "cuda"
) -> Tuple[torch.Tensor, List[Tuple[int, float, float, float]]]:
    """
    采样过程中跟踪 FID、IS 和 CLIP Score 指标
    
    Args:
        sampler: 扩散采样器
        x_T: 初始噪声 (batch_size, C, H, W)
        fid_calculator: FID 计算器
        is_calculator: IS 计算器
        clip_calculator: CLIP Score 计算器
        real_features: 真实数据的 Inception 特征 (N, 2048)，用于计算 FID
        real_clip_features: 真实数据的 CLIP 特征 (M, dim)，用于计算 CLIP Score
        mu_real: 预计算的真实特征均值（可选，用于优化）
        sigma_real: 预计算的真实特征协方差（可选，用于优化）
        metric_interval: 每隔多少步计算一次指标
        device: 计算设备
    
    Returns:
        x_0: 最终生成的图像
        metric_history: [(timestep, fid_value, is_value, clip_value), ...] 每个记录点的指标值
    """
    metric_history = []
    x_t = x_T
    T = sampler.T
    
    with torch.no_grad():
        for time_step in tqdm(reversed(range(T)), desc="Sampling"):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = sampler.p_mean_variance(x_t=x_t, t=t)
            
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            
            # 每隔 metric_interval 步计算一次 FID、IS 和 CLIP Score
            if time_step % metric_interval == 0 or time_step == 0:
                # 将中间图像转换为 [0, 1] 范围的 tensor
                x_denormalized = torch.clamp((x_t * 0.5 + 0.5), 0, 1)
                
                fid_value = float('nan')
                is_value = float('nan')
                clip_value = float('nan')
                
                try:
                    # 计算 FID（优化：使用预计算的real_features统计量）
                    fake_features = fid_calculator.extract_features_from_tensor(x_denormalized)
                    
                    # 如果未提供预计算的统计量，则计算（首次调用时）
                    if mu_real is None:
                        mu_real = real_features.mean(dim=0)
                        if real_features.shape[0] > 1:
                            sigma_real = torch.cov(real_features.T, correction=0)
                        else:
                            sigma_real = torch.eye(real_features.shape[1], dtype=torch.float64, device=real_features.device) * 1e-6
                        mu_real = mu_real.to(device)
                        sigma_real = sigma_real.to(device)
                    
                    # 计算fake_features的统计量（保持在GPU上）
                    mu_fake = fake_features.mean(dim=0)
                    if fake_features.shape[0] > 1:
                        sigma_fake = torch.cov(fake_features.T, correction=0)
                    else:
                        sigma_fake = torch.eye(fake_features.shape[1], dtype=torch.float64, device=fake_features.device) * 1e-6
                    
                    fid_value = fid_calculator.calculate_frechet_distance(
                        mu_real, sigma_real, mu_fake, sigma_fake
                    )
                except Exception as e:
                    print(f"Warning: FID calculation failed at step {time_step}: {e}")
                
                try:
                    # 计算 IS
                    is_mean, is_std = is_calculator.compute_is(x_denormalized)
                    is_value = is_mean
                except Exception as e:
                    print(f"Warning: IS calculation failed at step {time_step}: {e}")
                
                try:
                    # 计算 CLIP Score
                    if clip_calculator is not None:
                        fake_clip_features = clip_calculator.extract_features_from_tensor(x_denormalized)
                        clip_value = clip_calculator.compute_clip_score_with_features(
                            real_clip_features, fake_clip_features
                        )
                    else:
                        clip_value = float('nan')
                except Exception as e:
                    print(f"Warning: CLIP Score calculation failed at step {time_step}: {e}")
                    clip_value = float('nan')
                
                metric_history.append((time_step, fid_value, is_value, clip_value))
                if time_step % (metric_interval * 5) == 0:
                    print(f"Step {time_step}: FID={fid_value:.2f}, IS={is_value:.2f}, CLIP={clip_value:.4f}")
        
        x_0 = torch.clamp(x_t, -1, 1)
        return x_0, metric_history


def compute_real_features_for_fid(dataloader: DataLoader, fid_calculator: FID, 
                                   num_samples: int = 5000, device: str = "cuda") -> torch.Tensor:
    """
    从真实数据集中提取 Inception 特征，用于 FID 计算
    
    Args:
        dataloader: 数据加载器
        fid_calculator: FID 计算器
        num_samples: 需要提取的样本数量
        device: 计算设备
    
    Returns:
        features: (N, 2048) 特征张量
    """
    all_features = []
    count = 0
    
    print(f"Computing real features for FID (target: {num_samples} samples)...")
    with torch.no_grad():
        for images, _ in dataloader:
            if count >= num_samples:
                break
            images = images.to(device)
            # 将 [-1, 1] 归一化的图像转换为 [0, 1]
            images_denorm = torch.clamp((images * 0.5 + 0.5), 0, 1)
            
            # 批量提取特征
            batch_features = fid_calculator.extract_features_from_tensor(images_denorm)
            all_features.append(batch_features.cpu())
            count += images.shape[0]
            
            if count % 1000 == 0:
                print(f"  Processed {count} samples...")
    
    features = torch.cat(all_features[:num_samples], dim=0)
    print(f"Computed {features.shape[0]} real features.")
    return features


def compute_real_features_for_clip(dataloader: DataLoader, clip_calculator: CLIPScore, 
                                   num_samples: int = 5000, device: str = "cuda") -> torch.Tensor:
    """
    从真实数据集中提取 CLIP 特征，用于 CLIP Score 计算
    
    Args:
        dataloader: 数据加载器
        clip_calculator: CLIP Score 计算器
        num_samples: 需要提取的样本数量
        device: 计算设备
    
    Returns:
        features: (N, dim) CLIP 特征张量，已 L2 归一化
    """
    all_features = []
    count = 0
    
    print(f"Computing real CLIP features (target: {num_samples} samples)...")
    with torch.no_grad():
        for images, _ in dataloader:
            if count >= num_samples:
                break
            images = images.to(device)
            # 将 [-1, 1] 归一化的图像转换为 [0, 1]
            images_denorm = torch.clamp((images * 0.5 + 0.5), 0, 1)
            
            # 批量提取特征
            batch_features = clip_calculator.extract_features_from_tensor(images_denorm)
            all_features.append(batch_features.cpu())
            count += images.shape[0]
            
            if count % 1000 == 0:
                print(f"  Processed {count} samples...")
    
    features = torch.cat(all_features[:num_samples], dim=0)
    print(f"Computed {features.shape[0]} real CLIP features.")
    return features


def plot_loss_curve(loss_history: List[Tuple[int, float]], epoch: int, save_dir: str, 
                    smooth_window: int = 50, max_points: int = 2000):
    """
    绘制训练 loss-step 曲线图
    
    Args:
        loss_history: [(step, loss_value), ...]
        epoch: 当前 epoch
        save_dir: 保存目录
        smooth_window: 移动平均窗口大小（用于平滑）
        max_points: 最大显示点数（如果点数太多会降采样）
    """
    if len(loss_history) == 0:
        print(f"Warning: No loss values for epoch {epoch}")
        return
    
    steps = [s for s, _ in loss_history]
    losses = [l for _, l in loss_history]
    
    # Downsample if too many points
    if len(losses) > max_points:
        indices = np.linspace(0, len(losses) - 1, max_points, dtype=int)
        steps = [steps[i] for i in indices]
        losses = [losses[i] for i in indices]
    
    # Apply moving average smoothing if window size > 1
    if smooth_window > 1 and len(losses) > smooth_window:
        smoothed_losses = []
        for i in range(len(losses)):
            start_idx = max(0, i - smooth_window // 2)
            end_idx = min(len(losses), i + smooth_window // 2 + 1)
            smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
        losses_smooth = smoothed_losses
    else:
        losses_smooth = losses
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    # Plot original and smoothed
    plt.plot(steps, losses, linestyle='-', linewidth=0.8, color='blue', alpha=0.3, label='Raw')
    if smooth_window > 1 and len(losses) > smooth_window:
        plt.plot(steps, losses_smooth, linestyle='-', linewidth=1.5, color='red', alpha=0.8, label='Smoothed')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss vs Steps - Epoch {epoch}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    mean_loss = np.mean(losses)
    min_loss = np.min(losses)
    max_loss = np.max(losses)
    final_loss = losses[-1]
    plt.text(0.02, 0.98, 
             f'Mean: {mean_loss:.4f}\nMin: {min_loss:.4f}\nMax: {max_loss:.4f}\nFinal: {final_loss:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_path = os.path.join(save_dir, f'loss_curve_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def plot_metrics_curves(metric_history: List[Tuple[int, float, float, float]], epoch: int, save_dir: str):
    """
    绘制 FID-step、IS-step 和 CLIP Score-step 曲线图
    
    Args:
        metric_history: [(timestep, fid_value, is_value, clip_value), ...]
        epoch: 当前 epoch
        save_dir: 保存目录
    """
    steps = [t for t, _, _, _ in metric_history]
    fids = [f for _, f, _, _ in metric_history]
    iss = [i for _, _, i, _ in metric_history]
    clips = [c for _, _, _, c in metric_history]
    
    # 过滤掉 NaN 值（至少有一个有效值即可）
    valid_indices = [
        i for i, (_, f, is_val, c_val) in enumerate(metric_history)
        if not (isinstance(f, float) and np.isnan(f)) 
        or not (isinstance(is_val, float) and np.isnan(is_val))
        or not (isinstance(c_val, float) and np.isnan(c_val))
    ]
    
    if len(valid_indices) == 0:
        print(f"Warning: No valid metric values for epoch {epoch}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制 FID 曲线
    fid_indices = [i for i in valid_indices if not (isinstance(fids[i], float) and np.isnan(fids[i]))]
    if len(fid_indices) > 0:
        steps_fid = [steps[i] for i in fid_indices]
        fids_valid = [fids[i] for i in fid_indices]
        plt.figure(figsize=(10, 6))
        plt.plot(steps_fid, fids_valid, marker='o', linestyle='-', linewidth=2, markersize=4, color='blue')
        plt.xlabel('Diffusion Timestep (reverse)', fontsize=12)
        plt.ylabel('FID Score', fontsize=12)
        plt.title(f'FID vs Diffusion Steps - Epoch {epoch}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()  # 反转 x 轴，使得 t=0 在右边（去噪完成）
        save_path = os.path.join(save_dir, f'fid_curve_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"FID curve saved to: {save_path}")
    
    # 绘制 IS 曲线
    is_indices = [i for i in valid_indices if not (isinstance(iss[i], float) and np.isnan(iss[i]))]
    if len(is_indices) > 0:
        steps_is = [steps[i] for i in is_indices]
        iss_valid = [iss[i] for i in is_indices]
        plt.figure(figsize=(10, 6))
        plt.plot(steps_is, iss_valid, marker='s', linestyle='-', linewidth=2, markersize=4, color='green')
        plt.xlabel('Diffusion Timestep (reverse)', fontsize=12)
        plt.ylabel('IS Score', fontsize=12)
        plt.title(f'IS vs Diffusion Steps - Epoch {epoch}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        save_path = os.path.join(save_dir, f'is_curve_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"IS curve saved to: {save_path}")
    
    # 绘制 CLIP Score 曲线
    clip_indices = [i for i in valid_indices if not (isinstance(clips[i], float) and np.isnan(clips[i]))]
    if len(clip_indices) > 0:
        steps_clip = [steps[i] for i in clip_indices]
        clips_valid = [clips[i] for i in clip_indices]
        plt.figure(figsize=(10, 6))
        plt.plot(steps_clip, clips_valid, marker='^', linestyle='-', linewidth=2, markersize=4, color='red')
        plt.xlabel('Diffusion Timestep (reverse)', fontsize=12)
        plt.ylabel('CLIP Score', fontsize=12)
        plt.title(f'CLIP Score vs Diffusion Steps - Epoch {epoch}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        save_path = os.path.join(save_dir, f'clip_curve_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"CLIP Score curve saved to: {save_path}")


def train(cfg: Dict):
    # Setup multi-GPU support
    device_str = cfg.get("device", "cuda")
    use_multi_gpu = cfg.get("use_multi_gpu", True)
    
    # Get available GPU count (respects CUDA_VISIBLE_DEVICES)
    available_gpu_count = torch.cuda.device_count()
    
    # Parse device string
    if device_str.startswith("cuda"):
        if "," in device_str:
            # User specified multiple devices like "cuda:8,9"
            # Note: If CUDA_VISIBLE_DEVICES is set, these will map to cuda:0, cuda:1
            requested_device_ids = [int(x.split(":")[-1]) for x in device_str.split(",")]
            # Check if requested devices exist
            if all(id < available_gpu_count for id in requested_device_ids):
                device_ids = requested_device_ids
                primary_device = torch.device(f"cuda:{device_ids[0]}")
            else:
                # If devices don't exist (e.g., CUDA_VISIBLE_DEVICES changed mapping),
                # use all available GPUs
                print(f"Warning: Requested devices {requested_device_ids} not all available.")
                print(f"Available {available_gpu_count} GPUs. Using all available GPUs.")
                device_ids = list(range(available_gpu_count)) if available_gpu_count > 1 else None
                primary_device = torch.device("cuda:0")
        else:
            device_ids = None
            primary_device = torch.device(device_str)
    else:
        device_ids = None
        primary_device = torch.device(device_str)
    
    # Auto-detect multiple GPUs if enabled and not explicitly specified
    if use_multi_gpu and device_ids is None and available_gpu_count > 1:
        device_ids = list(range(available_gpu_count))
        primary_device = torch.device("cuda:0")
        print(f"Auto-detected {len(device_ids)} GPUs: {device_ids}")
    elif device_ids and len(device_ids) > 1:
        print(f"Using {len(device_ids)} GPUs: {device_ids} (mapped to cuda:{device_ids[0]}-{device_ids[-1]})")
    else:
        device_ids = None
        print(f"Using single device: {primary_device}")
    
    device = primary_device
    
    # dataset - ImageNet
    # ImageNet requires root directory with 'train' and 'val' subdirectories
    # Each subdirectory should contain class folders with images
    # Format: root/train/class1/, root/train/class2/, etc.
    img_size = cfg.get("img_size", 256)  # ImageNet typically uses 224 or 256
    imagenet_root = cfg.get("imagenet_root", "./ImageNet")
    split = 'train' if cfg.get("train", True) else 'val'
    
    # Use ImageFolder (works with standard ImageNet directory structure)
    # ImageFolder is more flexible and doesn't require devkit files
    train_dir = os.path.join(imagenet_root, split)
    
    # Check if directory exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"ImageNet dataset directory not found: {train_dir}\n"
            f"Please set 'imagenet_root' in your config to point to your ImageNet dataset.\n"
            f"Expected structure: {imagenet_root}/train/class1/, {imagenet_root}/train/class2/, etc."
        )
    
    # Use ImageFolder directly (simpler and more compatible)
    dataset = ImageFolder(
        root=train_dir,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    print(f"Loaded ImageNet dataset from {train_dir}")
    print(f"Total samples: {len(dataset)}, Number of classes: {len(dataset.classes)}")
    
    # Adjust batch size for multi-GPU (optional, can keep original batch size)
    effective_batch_size = cfg["batch_size"]
    if device_ids and len(device_ids) > 1:
        # Optionally scale batch size
        # effective_batch_size = cfg["batch_size"] * len(device_ids)
        pass
    
    dataloader = DataLoader(
        dataset, batch_size=effective_batch_size, shuffle=True, 
        num_workers=cfg.get("num_workers", 4), drop_last=True, pin_memory=True)
    
    # Create validation dataset for evaluation metrics
    val_dir = os.path.join(imagenet_root, 'val')
    use_val_for_eval = cfg.get("use_val_for_eval", True)  # Use validation set for evaluation metrics
    
    if use_val_for_eval and os.path.exists(val_dir):
        print(f"Creating validation dataset from {val_dir} for evaluation metrics...")
        val_dataset = ImageFolder(
            root=val_dir,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # No random flip for validation
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        print(f"Validation dataset: {len(val_dataset)} samples")
        eval_dataloader = DataLoader(
            val_dataset, batch_size=effective_batch_size, shuffle=False, 
            num_workers=cfg.get("num_workers", 4), drop_last=False, pin_memory=True)
    else:
        print("Using training set for evaluation metrics (validation set not found or disabled)")
        eval_dataloader = None

    # model setup
    net_model = UNet(T=cfg["T"], ch=cfg["channel"], ch_mult=cfg["channel_mult"], attn=cfg["attn"],
                     num_res_blocks=cfg["num_res_blocks"], dropout=cfg["dropout"]).to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if device_ids and len(device_ids) > 1:
        # Set output_device to the primary device (first GPU)
        net_model = torch.nn.DataParallel(
            net_model, 
            device_ids=device_ids,
            output_device=device_ids[0]  # Explicitly set output device
        )
        print(f"Model wrapped with DataParallel on devices {device_ids}, output on device {device_ids[0]}")
        # Get the actual model for saving/loading
        model_for_saving = net_model.module
    else:
        model_for_saving = net_model

    if cfg["training_load_weight"] is not None:
        checkpoint_path = os.path.join(
            cfg["save_weight_dir"], cfg["training_load_weight"])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both DataParallel and normal model state_dict
        # DataParallel saves keys with "module." prefix, unwrapped model doesn't
        state_dict = checkpoint
        if any(key.startswith("module.") for key in state_dict.keys()):
            # Remove "module." prefix if present (from DataParallel)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model_for_saving.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    #build optimizer, scheduler, warmupscheduler, trainer
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=cfg["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=cfg["multiplier"], warm_epoch=cfg["epoch"] // 10, after_scheduler=cosineScheduler)
    
    # Create trainer - use the wrapped model for training
    # Note: If net_model is DataParallel, trainer.model will be DataParallel
    # Trainer's buffers will be on the primary device, which is correct
    trainer = GaussianDiffusionTrainer(
        net_model, cfg["beta_1"], cfg["beta_T"], cfg["T"])
    
    # Move trainer to primary device (buffers like betas will be on primary device)
    # The model inside trainer is already DataParallel and will handle multi-GPU automatically
    trainer = trainer.to(device)
    
    # Verify DataParallel is working
    if device_ids and len(device_ids) > 1:
        print(f"Verifying DataParallel setup:")
        print(f"  - Trainer device: {next(trainer.parameters()).device}")
        print(f"  - Model is DataParallel: {isinstance(trainer.model, torch.nn.DataParallel)}")
        if isinstance(trainer.model, torch.nn.DataParallel):
            print(f"  - DataParallel device_ids: {trainer.model.device_ids}")
            print(f"  - DataParallel output_device: {trainer.model.output_device}")
    
    # Initialize FID, IS and CLIP Score calculators
    print("Initializing FID, IS and CLIP Score calculators...")
    fid_calculator = FID(device=device)
    is_calculator = IS(device=device)
    
    # Try to initialize CLIP Score calculator (may fail if CLIP is not installed)
    try:
        clip_calculator = CLIPScore(device=device)
        use_clip = True
        print("CLIP Score calculator initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize CLIP Score calculator: {e}")
        print("Continuing without CLIP Score...")
        clip_calculator = None
        use_clip = False
    
    # Pre-compute real features for FID using evaluation dataset (validation set if available)
    # This will be recomputed at each evaluation epoch if using validation set
    eval_data_source = eval_dataloader if eval_dataloader is not None else dataloader
    eval_data_name = "validation set" if eval_dataloader is not None else "training set"
    
    # Decide whether to pre-compute or compute on-the-fly
    # If using validation set, we can either pre-compute once or compute each epoch
    # For efficiency, we'll pre-compute once if not using validation set, or compute each epoch if using validation set
    precompute_real_features = cfg.get("precompute_real_features", not use_val_for_eval or eval_dataloader is None)
    
    if precompute_real_features:
        print(f"Pre-computing real features for FID from {eval_data_name}...")
        real_features = compute_real_features_for_fid(
            eval_data_source, fid_calculator, 
            num_samples=cfg.get("fid_num_real_samples", 5000), 
            device=device
        )
        
        # Pre-compute real_features statistics for FID (optimization: compute once)
        print("Pre-computing real features statistics for FID...")
        real_features_gpu = real_features.to(device)
        mu_real = real_features_gpu.mean(dim=0)
        if real_features_gpu.shape[0] > 1:
            sigma_real = torch.cov(real_features_gpu.T, correction=0)
        else:
            sigma_real = torch.eye(real_features_gpu.shape[1], dtype=torch.float64, device=device) * 1e-6
        print("Real features statistics pre-computed.")
        
        # Pre-compute real CLIP features if CLIP is available
        real_clip_features = None
        if use_clip:
            print(f"Pre-computing real CLIP features from {eval_data_name}...")
            real_clip_features = compute_real_features_for_clip(
                eval_data_source, clip_calculator,
                num_samples=cfg.get("clip_num_real_samples", 5000),
                device=device
            )
            # Move to GPU for faster access
            real_clip_features = real_clip_features.to(device)
    else:
        print(f"Will compute real features from {eval_data_name} at each evaluation epoch")
        real_features = None
        real_features_gpu = None
        mu_real = None
        sigma_real = None
        real_clip_features = None
    
    # Create sampler for evaluation - use unwrapped model for sampling
    # Sampling is sequential and doesn't benefit from DataParallel
    sampler = GaussianDiffusionSampler(
        model_for_saving, cfg["beta_1"], cfg["beta_T"], cfg["T"]).to(device)
    
    # Create metrics save directory
    metrics_save_dir = cfg.get("metrics_save_dir", "./metrics_curves")
    os.makedirs(metrics_save_dir, exist_ok=True)

    # start training
    for e in tqdm(range(cfg["epoch"]), desc="Training"):
        start_time = time.time()
        loss_history = []  # Record loss for this epoch
        global_step = e * len(dataloader)  # Calculate global step number
        
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch_idx, (images, labels) in enumerate(tqdmDataLoader):
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                # Compute MSE loss per element, then take mean (standard DDPM approach)
                # trainer(x_0) returns loss with shape [batch_size, C, H, W] with reduction='none'
                loss = trainer(x_0).mean()  # Take mean over all elements (standard practice)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), cfg["grad_clip"])
                optimizer.step()
                
                # Record loss
                current_step = global_step + batch_idx
                loss_value = loss.item()
                loss_history.append((current_step, loss_value))
                
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss_value,
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        save_path = os.path.join(cfg["save_weight_dir"], 'ckpt_' + str(e) + "_.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        warmUpScheduler.step()
        if e % 5 == 0 or e == cfg["epoch"] - 1:
            # Save the unwrapped model state_dict
            torch.save(model_for_saving.state_dict(), save_path)
        
        # Plot training loss curve (only every few epochs to save time)
        eval_freq = cfg.get("eval_freq", 5)  # Evaluate every N epochs (default: 5)
        if e % eval_freq == 0 or e == cfg["epoch"] - 1:
            print(f"\nPlotting training loss curve for epoch {e}...")
            plot_loss_curve(loss_history, epoch=e, save_dir=metrics_save_dir)
        
        # Evaluate metrics only at specified intervals (optimization: reduce frequency)
        if e % eval_freq == 0 or e == cfg["epoch"] - 1:
            print(f"\nEvaluating metrics for epoch {e}...")
            net_model.eval()
            # Update sampler with current unwrapped model (for sequential sampling)
            sampler.model = model_for_saving
            
            # Compute or use pre-computed real features for evaluation
            # If using validation set and not pre-computed, compute on-the-fly
            if mu_real is None or sigma_real is None:
                print(f"Computing real features from {eval_data_name} for this evaluation...")
                eval_real_features = compute_real_features_for_fid(
                    eval_data_source, fid_calculator, 
                    num_samples=cfg.get("fid_num_real_samples", 5000), 
                    device=device
                )
                eval_real_features_gpu = eval_real_features.to(device)
                eval_mu_real = eval_real_features_gpu.mean(dim=0)
                if eval_real_features_gpu.shape[0] > 1:
                    eval_sigma_real = torch.cov(eval_real_features_gpu.T, correction=0)
                else:
                    eval_sigma_real = torch.eye(eval_real_features_gpu.shape[1], dtype=torch.float64, device=device) * 1e-6
                
                eval_real_clip_features_eval = None
                if use_clip:
                    eval_real_clip_features = compute_real_features_for_clip(
                        eval_data_source, clip_calculator,
                        num_samples=cfg.get("clip_num_real_samples", 5000),
                        device=device
                    )
                    eval_real_clip_features_eval = eval_real_clip_features.to(device)
            else:
                # Use pre-computed features
                eval_real_features_gpu = real_features_gpu
                eval_mu_real = mu_real
                eval_sigma_real = sigma_real
                eval_real_clip_features_eval = real_clip_features if use_clip else None
            
            # Sample images with metrics tracking
            eval_batch_size = cfg.get("eval_batch_size")
            if eval_batch_size is None:
                eval_batch_size = min(cfg["batch_size"], 64)  # Limit eval batch size for speed
            
            img_size = cfg.get("img_size", 256)  # ImageNet default size
            x_T_eval = torch.randn(size=[eval_batch_size, 3, 
                                         img_size, 
                                         img_size], device=device)
            
            # Use larger metric_interval during evaluation to speed up
            eval_metric_interval = cfg.get("eval_metric_interval", cfg.get("metric_interval", 20))
            
            sampled_imgs, metric_history = sample_with_metrics_tracking(
                sampler=sampler,
                x_T=x_T_eval,
                fid_calculator=fid_calculator,
                is_calculator=is_calculator,
                clip_calculator=clip_calculator if use_clip else None,
                real_features=eval_real_features_gpu,
                real_clip_features=eval_real_clip_features_eval if use_clip else torch.zeros((eval_batch_size, 512), device=device),
                mu_real=eval_mu_real,  # Use computed or pre-computed statistics
                sigma_real=eval_sigma_real,  # Use computed or pre-computed statistics
                metric_interval=eval_metric_interval,  # Use larger interval for speed
                device=device
            )
            
            # Plot metrics curves
            plot_metrics_curves(metric_history, epoch=e, save_dir=metrics_save_dir)
            
            # Save sampled images
            sampled_imgs_denorm = sampled_imgs * 0.5 + 0.5  # [0 ~ 1]

            save_image_path = os.path.join(cfg.get("sampled_dir", "./SampledImgs"), f"epoch_{e}_sampled.png")
            os.makedirs(os.path.dirname(save_image_path),exist_ok=True)
            save_image(sampled_imgs_denorm, save_image_path, nrow=cfg.get("nrow", 8))
            print(f"Saved sampled images to {save_image_path}")
            
            net_model.train()  # Set back to training mode
        else:
            print(f"Skipping evaluation for epoch {e} (evaluate every {eval_freq} epochs)")
        
        print(f"Epoch {e} completed, LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}, Loss: {loss.item():.6f}, Time: {time.time() - start_time:.2f}s")


def eval(cfg: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(cfg["device"])
        model = UNet(T=cfg["T"], ch=cfg["channel"], ch_mult=cfg["channel_mult"], attn=cfg["attn"],
                     num_res_blocks=cfg["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            cfg["save_weight_dir"], cfg["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, cfg["beta_1"], cfg["beta_T"], cfg["T"]).to(device)
        # Sampled from standard normal distribution
        img_size = cfg.get("img_size", 256)  # ImageNet default size
        noisyImage = torch.randn(
            size=[cfg["batch_size"], 3, img_size, img_size], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        sampled_dir = cfg["sampled_dir"]
        os.makedirs(sampled_dir, exist_ok=True)
        noisy_img_path = os.path.join(sampled_dir, cfg["sampledNoisyImgName"])
        save_image(saveNoisy, noisy_img_path, nrow=cfg["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        sampled_img_path = os.path.join(sampled_dir, cfg["sampledImgName"])
        save_image(sampledImgs, sampled_img_path, nrow=cfg["nrow"])