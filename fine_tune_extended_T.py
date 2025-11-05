"""
Fine-tuning script for extending T (diffusion steps) from checkpoint

This script loads a checkpoint trained with T=1000, extends it to T=2000 (or other T),
and fine-tunes only the time_embedding layers to adapt to the new T value.

Usage:
    python fine_tune_extended_T.py                    # Use default config
    python fine_tune_extended_T.py T=2000 epoch=5    # Override params
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, '/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-')
from Diffusion.Train import train
from Diffusion.Model import UNet
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="config", config_name="fine_tune_config")
def main(cfg: DictConfig) -> None:
    """
    Fine-tuning function for extended T
    """
    # Convert OmegaConf to dict
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
    
    # Override config for fine-tuning
    config["fine_tune_mode"] = True
    config["fine_tune_time_embedding"] = True
    
    # Print configuration
    print("=" * 80)
    print("Fine-tuning Configuration (Extended T):")
    print("=" * 80)
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Modify train function to handle fine-tuning
    # We'll need to patch the model loading in train function
    # For now, let's create a wrapper that handles the extension
    
    print("\n" + "=" * 80)
    print("Starting Fine-tuning for Extended T")
    print("=" * 80)
    print(f"Original checkpoint T: {config.get('checkpoint_T', 'unknown')}")
    print(f"Target T: {config['T']}")
    print(f"Will fine-tune only time_embedding layers")
    print("=" * 80)
    
    # Call the modified train function
    train_with_extended_T(config)


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
    
    # Parse device IDs
    if device_ids is not None:
        if isinstance(device_ids, (list, tuple)):
            device_ids = list(device_ids)
        elif isinstance(device_ids, str):
            device_ids = [int(x.strip()) for x in device_ids.replace('[', '').replace(']', '').split(',')]
        else:
            device_ids = [int(device_ids)]
        
        if cuda_visible_devices:
            available_gpu_count = torch.cuda.device_count()
            requested_count = len(device_ids)
            if requested_count > available_gpu_count:
                device_ids = list(range(available_gpu_count))
            else:
                device_ids = list(range(requested_count))
            print(f"Remapped device IDs to: {device_ids}")
        
        primary_device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device_ids = None
        primary_device = torch.device(device_str)
    
    print(f"Using device: {primary_device}")
    if device_ids and len(device_ids) > 1:
        print(f"Will use DataParallel on devices: {device_ids}")
    
    return primary_device, device_ids


def load_dataset(config: dict) -> torch.utils.data.DataLoader:
    """
    Load ImageNet training dataset
    
    Returns:
        dataloader: DataLoader for training
    """
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    
    imagenet_root = config.get("imagenet_root", "./ImageNet")
    img_size = config.get("img_size", 256)
    train_dir = os.path.join(imagenet_root, 'train')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Dataset directory not found: {train_dir}")
    
    dataset = ImageFolder(
        root=train_dir,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config.get("num_workers", 4), 
        drop_last=True, 
        pin_memory=True
    )
    
    print(f"Loaded dataset: {len(dataset)} samples")
    return dataloader


def load_checkpoint_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """
    Load and extract state_dict from checkpoint
    
    Returns:
        state_dict: Model state dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please set 'checkpoint_path' in config to point to your checkpoint file."
        )
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            raise ValueError("Could not extract state_dict from checkpoint")
    
    # Handle DataParallel prefix
    if any(key.startswith("module.") for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    return state_dict


def detect_and_prepare_checkpoint_T(state_dict: dict, target_T: int) -> dict:
    """
    Detect checkpoint T and remove time_embedding if T mismatch
    
    Returns:
        state_dict: Modified state_dict with time_embedding removed if needed
    """
    checkpoint_T = None
    time_emb_key = "time_embedding.timembedding.0.weight"
    
    if time_emb_key in state_dict:
        checkpoint_T = state_dict[time_emb_key].shape[0]
        print(f"Detected checkpoint T={checkpoint_T}")
    
    # Remove time_embedding if T mismatch
    if checkpoint_T and checkpoint_T != target_T:
        print(f"\n⚠️  T mismatch: checkpoint T={checkpoint_T}, target T={target_T}")
        print("Removing time_embedding weights for extension...")
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("time_embedding")]
        for k in keys_to_remove:
            del state_dict[k]
            print(f"  Removed: {k}")
    
    return state_dict


def create_and_load_model(config: dict, device: torch.device, device_ids: list, 
                         state_dict: dict) -> tuple:
    """
    Create model, wrap with DataParallel if needed, and load weights
    
    Returns:
        model_for_training: Model for training (may be DataParallel)
        model_for_access: Unwrapped model for accessing parameters
    """
    # Create model with extended T
    print("\nCreating model with extended T...")
    model = UNet(
        T=config["T"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"]
    ).to(device)
    
    # Wrap model with DataParallel if multiple GPUs
    if device_ids and len(device_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0]
        )
        print(f"Model wrapped with DataParallel on devices {device_ids}")
        model_for_training = model
        model_for_access = model.module
    else:
        model_for_training = model
        model_for_access = model
    
    # Load weights (strict=False to allow missing time_embedding)
    missing_keys, unexpected_keys = model_for_access.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    return model_for_training, model_for_access


def freeze_parameters_except_time_embedding(model: nn.Module) -> list:
    """
    Freeze all parameters except time_embedding layers
    
    Returns:
        trainable_params: List of trainable parameters
    """
    print("\nFreezing all parameters except time_embedding...")
    for name, param in model.named_parameters():
        if 'time_embedding' in name:
            param.requires_grad = True
            print(f"  Trainable: {name}")
        else:
            param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
    print(f"Total parameters: {total_count:,}")
    
    return trainable_params


def setup_optimizer_and_scheduler(config: dict, trainable_params: list) -> tuple:
    """
    Setup optimizer and learning rate scheduler
    
    Returns:
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
    """
    import torch.optim as optim
    from Scheduler import GradualWarmupScheduler
    
    fine_tune_lr = config.get("fine_tune_lr", config.get("lr", 1e-4) * 0.1)
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=fine_tune_lr,
        weight_decay=1e-4
    )
    print(f"Using learning rate: {fine_tune_lr}")
    
    num_epochs = config.get("fine_tune_epochs", config.get("epoch", 5))
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=config.get("multiplier", 2.0), 
        warm_epoch=max(1, num_epochs // 10), 
        after_scheduler=cosineScheduler
    )
    
    return optimizer, warmUpScheduler, num_epochs


def train_one_epoch(model: nn.Module, trainer, dataloader, optimizer: torch.optim.Optimizer,
                   trainable_params: list, device: torch.device, epoch: int, 
                   grad_clip: float) -> float:
    """
    Train for one epoch
    
    Returns:
        avg_loss: Average loss for this epoch
    """
    from tqdm import tqdm
    
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    with tqdm(dataloader, dynamic_ncols=True, leave=False) as tqdmDataLoader:
        for batch_idx, (images, labels) in enumerate(tqdmDataLoader):
            optimizer.zero_grad()
            x_0 = images.to(device)
            
            # Compute loss
            loss = trainer(x_0).mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            tqdmDataLoader.set_postfix({
                "epoch": epoch,
                "loss": f"{loss.item():.6f}",
                "LR": f"{optimizer.state_dict()['param_groups'][0]['lr']:.6e}"
            })
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def save_checkpoint(model: nn.Module, save_dir: str, epoch: int, T: int, 
                   save_freq: int, is_final: bool = False):
    """
    Save model checkpoint
    """
    if (epoch + 1) % save_freq == 0 or is_final:
        save_path = os.path.join(save_dir, f'fine_tuned_T{T}_epoch_{epoch+1}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\nSaved checkpoint: {save_path}")


def train_with_extended_T(config: dict):
    """
    Main fine-tuning function that handles extended T fine-tuning
    """
    from tqdm import tqdm
    import time
    from Diffusion import GaussianDiffusionTrainer
    
    # Setup device
    device, device_ids = setup_device(config)
    
    # Load dataset
    dataloader = load_dataset(config)
    
    # Load checkpoint
    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        training_load_weight = config.get("training_load_weight")
        if training_load_weight:
            checkpoint_path = os.path.join(config["save_weight_dir"], training_load_weight)
    
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    
    # Prepare state_dict for extended T
    state_dict = detect_and_prepare_checkpoint_T(state_dict, config["T"])
    
    # Create and load model
    model_for_training, model_for_access = create_and_load_model(
        config, device, device_ids, state_dict
    )
    
    # Freeze parameters except time_embedding
    trainable_params = freeze_parameters_except_time_embedding(model_for_access)
    
    # Setup optimizer and scheduler
    optimizer, scheduler, num_epochs = setup_optimizer_and_scheduler(config, trainable_params)
    
    # Setup trainer
    trainer = GaussianDiffusionTrainer(
        model_for_training, config["beta_1"], config["beta_T"], config["T"]
    ).to(device)
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Fine-tuning Training")
    print("=" * 80)
    
    for e in tqdm(range(num_epochs), desc="Fine-tuning"):
        start_time = time.time()
        
        # Train one epoch
        avg_loss = train_one_epoch(
            model_for_training, trainer, dataloader, optimizer,
            trainable_params, device, e, config.get("grad_clip", 1.0)
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        save_checkpoint(
            model_for_access, config["save_weight_dir"], e, config["T"],
            config.get("model_save_freq", 1), is_final=(e == num_epochs - 1)
        )
        
        print(f"Epoch {e+1}/{num_epochs}: Loss={avg_loss:.6f}, "
              f"LR={optimizer.state_dict()['param_groups'][0]['lr']:.6e}, "
              f"Time={time.time()-start_time:.2f}s")
    
    print("\n" + "=" * 80)
    print("Fine-tuning completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

