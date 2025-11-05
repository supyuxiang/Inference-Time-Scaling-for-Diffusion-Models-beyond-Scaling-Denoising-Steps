import torch
import torchvision
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import sys
sys.path.insert(0, '/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-')
from Diffusion.Train import sample_with_metrics_tracking, compute_real_features_for_fid, compute_real_features_for_clip, plot_metrics_curves
from utils.metrics import FID, IS, CLIPScore
from Diffusion import GaussianDiffusionSampler
from Diffusion.Model import UNet
from omegaconf import DictConfig, OmegaConf
import hydra


def parse_config(cfg: DictConfig) -> dict:
    """
    Parse and normalize configuration from Hydra
    
    Returns:
        config: Normalized configuration dictionary
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Convert None/null strings and boolean strings to actual types
    for key, value in config.items():
        if isinstance(value, str):
            if value.lower() == "none" or value.lower() == "null":
                config[key] = None
            elif value.lower() == "true":
                config[key] = True
            elif value.lower() == "false":
                config[key] = False
    
    return config


def print_config(config: dict):
    """Print configuration in a formatted way"""
    print("=" * 80)
    print("Inference Configuration:")
    print("=" * 80)
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("=" * 80)


def setup_device(config: dict) -> tuple:
    """
    Setup device and multi-GPU configuration
    
    Returns:
        device: Primary device
        device_ids: List of device IDs for DataParallel (None if single GPU)
    """
    device_str = config.get("device", "cuda")
    use_multi_gpu = config.get("use_multi_gpu", False)
    device_ids = config.get("device_ids", None)
    
    # Check if CUDA_VISIBLE_DEVICES is set
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
        print("Note: PyTorch will remap visible GPUs to 0, 1, 2, ...")
        available_gpu_count = torch.cuda.device_count()
        print(f"PyTorch sees {available_gpu_count} GPU(s)")
    
    # Parse device string or device_ids
    if device_ids is not None:
        # Use specified device IDs
        if isinstance(device_ids, (list, tuple)):
            device_ids = list(device_ids)
        elif isinstance(device_ids, str):
            device_ids = [int(x.strip()) for x in device_ids.replace('[', '').replace(']', '').split(',')]
        else:
            device_ids = [int(device_ids)]
        
        # If CUDA_VISIBLE_DEVICES is set, remap device IDs to 0, 1, ...
        if cuda_visible_devices:
            available_gpu_count = torch.cuda.device_count()
            requested_count = len(device_ids)
            if requested_count > available_gpu_count:
                print(f"Warning: Requested {requested_count} GPUs but only {available_gpu_count} available.")
                print(f"Using all available GPUs: {list(range(available_gpu_count))}")
                device_ids = list(range(available_gpu_count))
            else:
                device_ids = list(range(requested_count))
                print(f"Remapped device IDs to: {device_ids} (CUDA_VISIBLE_DEVICES={cuda_visible_devices})")
        else:
            # Check if devices exist
            available_gpu_count = torch.cuda.device_count()
            for dev_id in device_ids:
                if dev_id >= available_gpu_count:
                    raise RuntimeError(
                        f"Device {dev_id} does not exist. "
                        f"Available devices: 0-{available_gpu_count-1}. "
                        f"Use CUDA_VISIBLE_DEVICES to select specific GPUs."
                    )
        
        primary_device = torch.device(f"cuda:{device_ids[0]}")
        print(f"Using specified GPUs: {device_ids}")
    elif use_multi_gpu and device_str.startswith("cuda"):
        if "," in device_str:
            requested_ids = [int(x.split(":")[-1]) for x in device_str.split(",")]
            if cuda_visible_devices:
                device_ids = list(range(len(requested_ids)))
            else:
                device_ids = requested_ids
            primary_device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device_ids = list(range(torch.cuda.device_count()))
            primary_device = torch.device("cuda:0")
        print(f"Using GPUs: {device_ids}")
    else:
        device_ids = None
        primary_device = torch.device(device_str)
        print(f"Using single device: {primary_device}")
    
    return primary_device, device_ids


def load_checkpoint_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """
    Load and extract state_dict from checkpoint
    
    Returns:
        state_dict: Model state dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both state_dict and full model checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print("Warning: Checkpoint appears to be a full model. Trying to extract state_dict...")
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            raise ValueError("Could not extract state_dict from checkpoint")
    
    # Handle DataParallel prefix (module.)
    if any(key.startswith("module.") for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    return state_dict


def detect_checkpoint_T(state_dict: dict) -> int:
    """
    Detect checkpoint T from time_embedding weight shape.
    
    For old implementation (Embedding table): shape is [T, d_model]
    For new functional implementation (Linear layer): shape is [dim, d_model], so we can't detect T
    
    Returns:
        checkpoint_T if detected (old implementation), None otherwise (new functional implementation)
    """
    time_emb_key = "time_embedding.timembedding.0.weight"
    if time_emb_key in state_dict:
        weight_shape = state_dict[time_emb_key].shape
        # Old implementation: Embedding table with shape [T, d_model] where T could be 1000, 2000, etc.
        # New implementation: Linear layer with shape [dim, d_model] where dim is typically 512 or similar
        # If shape[0] is very large (> 1000), it's likely the old Embedding table
        # If shape[0] is small (< 1000), it's likely the new Linear layer
        if weight_shape[0] > 500:  # Likely old Embedding table (T >= 500)
            checkpoint_T = weight_shape[0]
            print(f"Detected checkpoint T={checkpoint_T} from old embedding table (shape {weight_shape})")
            return checkpoint_T
        else:
            # New functional implementation - T is not stored in weights
            print(f"Detected functional time embedding (Linear layer shape {weight_shape}) - T is not constrained")
            return None
    return None


def reinitialize_time_embedding(model: nn.Module, checkpoint_T: int, current_T: int, 
                               config: dict, device: torch.device):
    """
    Reinitialize time_embedding layer for extended T
    """
    import math
    from torch.nn import init as nn_init
    
    # Get target model (unwrap DataParallel if needed)
    if isinstance(model, torch.nn.DataParallel):
        target_model = model.module
    else:
        target_model = model
    
    # Get embedding dimensions
    embedding_layer = target_model.time_embedding.timembedding[0]
    if isinstance(embedding_layer, torch.nn.Embedding):
        d_model = embedding_layer.embedding_dim
    else:
        d_model = embedding_layer.weight.shape[1]
    
    # Strategy: Use interpolation/extrapolation instead of complete reinitialization
    embedding_strategy = config.get("time_embedding_strategy", "interpolate")
    
    if embedding_strategy == "interpolate" and checkpoint_T < current_T:
        print(f"Using interpolation strategy to extend time_embedding from T={checkpoint_T} to T={current_T}")
        
        # Create checkpoint embedding pattern
        emb_checkpoint_pattern = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb_checkpoint_pattern = torch.exp(-emb_checkpoint_pattern)
        pos_checkpoint = torch.arange(checkpoint_T).float()
        emb_checkpoint_base = pos_checkpoint[:, None] * emb_checkpoint_pattern[None, :]
        emb_checkpoint_base = torch.stack([torch.sin(emb_checkpoint_base), torch.cos(emb_checkpoint_base)], dim=-1)
        emb_checkpoint_base = emb_checkpoint_base.view(checkpoint_T, d_model)
        
        # Create new embedding for T=current_T
        emb_new_pattern = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb_new_pattern = torch.exp(-emb_new_pattern)
        pos_new = torch.arange(current_T).float()
        emb_new_base = pos_new[:, None] * emb_new_pattern[None, :]
        emb_new_base = torch.stack([torch.sin(emb_new_base), torch.cos(emb_new_base)], dim=-1)
        emb_new_base = emb_new_base.view(current_T, d_model)
        
        # Scale pattern for consistency
        emb = emb_new_base.clone()
        scale_factor = checkpoint_T / current_T
        emb[:checkpoint_T] = emb_new_base[:checkpoint_T] * scale_factor
        if current_T > checkpoint_T:
            emb[checkpoint_T:] = emb_new_base[checkpoint_T:]
        
        print(f"  Extended embedding: first {checkpoint_T} steps use checkpoint pattern, "
              f"remaining {current_T - checkpoint_T} steps use extrapolated pattern")
    else:
        # Complete reinitialization
        print(f"Using reinitialization strategy (strategy={embedding_strategy})")
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(current_T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(current_T, d_model)
    
    # Update embedding weights
    if isinstance(embedding_layer, torch.nn.Embedding):
        new_embedding = torch.nn.Embedding.from_pretrained(emb, freeze=False)
        target_model.time_embedding.timembedding[0] = new_embedding.to(device)
    else:
        embedding_layer.weight.data = emb.clone().to(device)
    
    # Handle linear layers
    fine_tune_time_embedding = config.get("fine_tune_time_embedding", False)
    if fine_tune_time_embedding:
        print("  Linear layers will be trainable (fine-tuning enabled)")
        for module in target_model.time_embedding.timembedding[1:]:
            if isinstance(module, torch.nn.Linear):
                for param in module.parameters():
                    param.requires_grad = True
    else:
        print("  Linear layers reinitialized (fine-tuning disabled)")
        for module in target_model.time_embedding.timembedding[1:]:
            if isinstance(module, torch.nn.Linear):
                nn_init.xavier_uniform_(module.weight)
                nn_init.zeros_(module.bias)
                for param in module.parameters():
                    param.requires_grad = False
    
    print(f"✅ Time embedding reinitialized for T={current_T}")


def create_and_load_model(config: dict, device: torch.device, device_ids: list) -> tuple:
    """
    Create model, wrap with DataParallel if needed, and load checkpoint
    
    Returns:
        model: Model (may be DataParallel)
        model_for_sampler: Unwrapped model for sampler
    """
    # Create model
    print("\nCreating model...")
    model = UNet(
        T=config["T"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"]
    ).to(device)
    
    # Wrap model with DataParallel if multiple GPUs are specified
    if device_ids and len(device_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0]
        )
        print(f"Model wrapped with DataParallel on devices {device_ids}, output on device {device_ids[0]}")
        model_for_sampler = model.module
    else:
        model_for_sampler = model
    
    # Load checkpoint
    checkpoint_path = config["checkpoint_path"]
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    
    # Detect checkpoint T
    checkpoint_T = detect_checkpoint_T(state_dict)
    current_T = config["T"]
    
    # Handle T mismatch
    if checkpoint_T is not None and checkpoint_T != current_T:
        print(f"\n⚠️  T mismatch detected: checkpoint T={checkpoint_T}, current config T={current_T}")
        print(f"Will load all weights except time_embedding, and reinitialize time_embedding for T={current_T}")
        
        # Remove time_embedding weights from state_dict
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("time_embedding")]
        for k in keys_to_remove:
            del state_dict[k]
            print(f"  Removed: {k}")
        
        # Load partial state_dict
        if isinstance(model, torch.nn.DataParallel):
            missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # Reinitialize time_embedding
        reinitialize_time_embedding(model, checkpoint_T, current_T, config, device)
        
        if missing_keys:
            print(f"Missing keys (expected for time_embedding): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    else:
        # Normal loading when T matches
        if isinstance(model, torch.nn.DataParallel):
            missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, model_for_sampler


def create_sampler(model: nn.Module, model_for_sampler: nn.Module, 
                  config: dict, device: torch.device) -> GaussianDiffusionSampler:
    """
    Create diffusion sampler
    
    Returns:
        sampler: GaussianDiffusionSampler instance
    """
    print("\nCreating diffusion sampler...")
    sampler = GaussianDiffusionSampler(
        model_for_sampler, config["beta_1"], config["beta_T"], config["T"]
    ).to(device)
    
    # Use DataParallel model if available for faster inference
    if isinstance(model, torch.nn.DataParallel):
        sampler.model = model
        print("Sampler using DataParallel model for faster inference")
    
    return sampler


def initialize_metrics_calculators(device: torch.device) -> tuple:
    """
    Initialize FID, IS, and CLIP Score calculators
    
    Returns:
        fid_calculator: FID calculator
        is_calculator: IS calculator
        clip_calculator: CLIP Score calculator (None if failed)
        use_clip: Whether CLIP is available
    """
    print("\nInitializing metrics calculators...")
    fid_calculator = FID(device=device)
    is_calculator = IS(device=device)
    
    # Try to initialize CLIP Score calculator
    try:
        clip_calculator = CLIPScore(device=device)
        use_clip = True
        print("CLIP Score calculator initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize CLIP Score calculator: {e}")
        print("Continuing without CLIP Score...")
        clip_calculator = None
        use_clip = False
    
    return fid_calculator, is_calculator, clip_calculator, use_clip


def load_eval_dataset(config: dict) -> DataLoader:
    """
    Load evaluation dataset (val or train split)
    
    Returns:
        dataloader: DataLoader for evaluation dataset
    """
    print("\nLoading dataset for real features...")
    imagenet_root = config["imagenet_root"]
    use_val = config.get("use_val_for_eval", True)
    split = 'val' if use_val else 'train'
    dataset_dir = os.path.join(imagenet_root, split)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please set 'imagenet_root' in your config."
        )
    
    dataset = ImageFolder(
        root=dataset_dir,
        transform=transforms.Compose([
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {len(dataset)} samples from {split} set")
    return dataloader


def compute_real_features(config: dict, dataloader: DataLoader, 
                          fid_calculator: FID, clip_calculator: CLIPScore,
                          use_clip: bool, device: torch.device) -> tuple:
    """
    Compute real features for FID and CLIP Score
    
    Returns:
        real_features_gpu: Real features for FID (on GPU)
        mu_real: Mean of real features
        sigma_real: Covariance of real features
        real_clip_features: Real CLIP features (on GPU, None if not using CLIP)
    """
    # Compute real features for FID
    print("\nComputing real features for FID...")
    real_features = compute_real_features_for_fid(
        dataloader,
        fid_calculator,
        num_samples=config.get("fid_num_real_samples", 5000),
        device=device
    )
    real_features_gpu = real_features.to(device)
    
    # Pre-compute real features statistics for FID
    print("Pre-computing real features statistics for FID...")
    mu_real = real_features_gpu.mean(dim=0)
    if real_features_gpu.shape[0] > 1:
        sigma_real = torch.cov(real_features_gpu.T, correction=0)
    else:
        sigma_real = torch.eye(real_features_gpu.shape[1], dtype=torch.float64, device=device) * 1e-6
    print("Real features statistics pre-computed.")
    
    # Compute real CLIP features if CLIP is available
    real_clip_features = None
    if use_clip:
        print("\nComputing real CLIP features...")
        real_clip_features = compute_real_features_for_clip(
            dataloader,
            clip_calculator,
            num_samples=config.get("clip_num_real_samples", 5000),
            device=device
        )
        real_clip_features = real_clip_features.to(device)
        print("Real CLIP features computed.")
    
    return real_features_gpu, mu_real, sigma_real, real_clip_features


def run_inference_with_metrics(sampler: GaussianDiffusionSampler, config: dict,
                               fid_calculator: FID, is_calculator: IS,
                               clip_calculator: CLIPScore, use_clip: bool,
                               real_features_gpu: torch.Tensor, mu_real: torch.Tensor,
                               sigma_real: torch.Tensor, real_clip_features: torch.Tensor,
                               device: torch.device) -> tuple:
    """
    Run inference with metrics tracking
    
    Returns:
        sampled_imgs: Generated images
        metric_history: List of (step, fid, is, clip) tuples
    """
    # Generate initial noise
    print("\nGenerating initial noise...")
    batch_size = config["batch_size"]
    img_size = config["img_size"]
    x_T = torch.randn(size=[batch_size, 3, img_size, img_size], device=device)
    
    # Sample with metrics tracking
    print("\nStarting inference with metrics tracking...")
    print(f"Metric interval: {config['metric_interval']} steps")
    
    sampled_imgs, metric_history = sample_with_metrics_tracking(
        sampler=sampler,
        x_T=x_T,
        fid_calculator=fid_calculator,
        is_calculator=is_calculator,
        clip_calculator=clip_calculator if use_clip else None,
        real_features=real_features_gpu,
        real_clip_features=real_clip_features if use_clip else torch.zeros((batch_size, 512), device=device),
        mu_real=mu_real,
        sigma_real=sigma_real,
        metric_interval=config["metric_interval"],
        device=device
    )
    
    return sampled_imgs, metric_history


def print_metrics_summary(metric_history: list, use_clip: bool):
    """Print final metrics summary"""
    print("\n" + "=" * 80)
    print("Final Metrics Summary:")
    print("=" * 80)
    if metric_history:
        final_step, final_fid, final_is, final_clip = metric_history[-1]
        print(f"Step {final_step}:")
        print(f"  FID: {final_fid:.4f}")
        print(f"  IS: {final_is:.4f}")
        if use_clip:
            print(f"  CLIP Score: {final_clip:.4f}")
        else:
            print(f"  CLIP Score: N/A (not enabled)")


def generate_image_filename(config: dict, metric_history: list = None) -> str:
    """
    Generate a descriptive filename for sampled images based on config and optional metrics
    
    Args:
        config: Configuration dictionary
        metric_history: Optional list of metric tuples (step, fid, is, clip)
    
    Returns:
        filename: Filename string without extension
    """
    from datetime import datetime
    
    # Extract checkpoint name from path
    checkpoint_path = config.get("checkpoint_path", "")
    checkpoint_name = "unknown"
    if checkpoint_path:
        # Try to extract meaningful name from checkpoint path
        # Example: /path/to/ep15_bs40_T1000_lr1e-4/ckpt_0_.pt -> ep15_bs40_T1000_lr1e-4
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_dir)
        # Clean up the name (keep underscores for readability)
        checkpoint_name = checkpoint_name.replace("Checkpoints/", "").replace("checkpoints/", "")
    
    # Get key parameters
    T = config.get("T", "unknown")
    batch_size = config.get("batch_size", "unknown")
    img_size = config.get("img_size", "unknown")
    
    # Build filename components
    filename_parts = [checkpoint_name, f"T{T}", f"bs{batch_size}", f"size{img_size}"]
    
    # Optionally add final metrics to filename if available
    if metric_history and len(metric_history) > 0:
        final_step, final_fid, final_is, final_clip = metric_history[-1]
        # Format FID to 2 decimal places for filename
        if final_fid is not None and not (isinstance(final_fid, float) and (final_fid != final_fid)):
            fid_str = f"fid{final_fid:.2f}".replace(".", "p")
            filename_parts.append(fid_str)
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts.append(timestamp)
    
    filename = "_".join(str(part) for part in filename_parts)
    # Sanitize filename (remove any problematic characters)
    filename = filename.replace("/", "-").replace("\\", "-").replace(":", "-")
    return filename


def save_results(sampled_imgs: torch.Tensor, metric_history: list, config: dict):
    """
    Save sampled images, plot metrics curves, and save metrics to JSON
    """
    # Plot metrics curves
    print("\nPlotting metrics curves...")
    metrics_save_dir = config.get("metrics_save_dir", "./inference_metrics_curves")
    os.makedirs(metrics_save_dir, exist_ok=True)
    plot_metrics_curves(metric_history, epoch=0, save_dir=metrics_save_dir)
    
    # Save sampled images
    print("\nSaving sampled images...")
    output_dir = config.get("output_dir", "./inference_results")
    os.makedirs(output_dir, exist_ok=True)
    
    sampled_imgs_denorm = sampled_imgs * 0.5 + 0.5  # [0 ~ 1]
    sampled_images_dir = config.get("sampled_images_save_dir", 
                                     os.path.join(output_dir, "sampled_images"))
    os.makedirs(sampled_images_dir, exist_ok=True)
    
    # Generate descriptive filename (include metric history for final FID)
    filename_base = generate_image_filename(config, metric_history)
    sampled_images_path = os.path.join(sampled_images_dir, f"{filename_base}.png")
    save_image(sampled_imgs_denorm, sampled_images_path, nrow=config.get("nrow", 8))
    print(f"Sampled images saved to: {sampled_images_path}")
    
    # Save metrics to file
    import json
    metrics_file = os.path.join(output_dir, "metrics_history.json")
    metrics_data = {
        "metric_history": [
            {
                "step": int(step),
                "fid": float(fid) if not (isinstance(fid, float) and (fid != fid)) else None,
                "is": float(is_val) if not (isinstance(is_val, float) and (is_val != is_val)) else None,
                "clip": float(clip) if not (isinstance(clip, float) and (clip != clip)) else None
            }
            for step, fid, is_val, clip in metric_history
        ]
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics history saved to: {metrics_file}")


@hydra.main(version_base=None, config_path="config", config_name="inference_config")
def main(cfg: DictConfig) -> None:
    """
    Main function for inference with metrics tracking
    
    Usage:
        python abstract_metrics_from_pretrained_ddpm.py                          # Use default config
        python abstract_metrics_from_pretrained_ddpm.py batch_size=32 metric_interval=20  # Override params
    """
    # Parse configuration
    config = parse_config(cfg)
    print_config(config)
    
    # Setup device
    device, device_ids = setup_device(config)
    
    # Create and load model
    model, model_for_sampler = create_and_load_model(config, device, device_ids)
    
    # Create sampler
    sampler = create_sampler(model, model_for_sampler, config, device)
    
    # Initialize metrics calculators
    fid_calculator, is_calculator, clip_calculator, use_clip = initialize_metrics_calculators(device)
    
    # Load evaluation dataset
    eval_dataloader = load_eval_dataset(config)
    
    # Compute real features
    real_features_gpu, mu_real, sigma_real, real_clip_features = compute_real_features(
        config, eval_dataloader, fid_calculator, clip_calculator, use_clip, device
    )
    
    # Run inference with metrics tracking
    sampled_imgs, metric_history = run_inference_with_metrics(
        sampler, config, fid_calculator, is_calculator, clip_calculator, use_clip,
        real_features_gpu, mu_real, sigma_real, real_clip_features, device
    )
    
    # Print summary and save results
    print_metrics_summary(metric_history, use_clip)
    save_results(sampled_imgs, metric_history, config)
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    # Set CUDA_VISIBLE_DEVICES if device_ids are specified via config
    # But if user wants to use cuda:4,5, they should set CUDA_VISIBLE_DEVICES=4,5
    # Or we can set it programmatically if device_ids is provided
    # Note: CUDA_VISIBLE_DEVICES must be set before importing torch
    # So this won't work if torch is already imported
    # Instead, we'll handle device selection in the main function
    
    main()





















