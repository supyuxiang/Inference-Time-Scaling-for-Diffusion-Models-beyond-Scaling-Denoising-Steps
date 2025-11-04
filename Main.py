from Diffusion.Train import train, eval
import sys
import os

sys.path.insert(0, '/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-')

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function using Hydra for configuration management
    
    Usage:
        python Main.py                          # Use default config
        python Main.py epoch=100 batch_size=512  # Override specific params
        python Main.py model_config.epoch=100    # Nested override (legacy format)
    """
    # Handle legacy nested config format (model_config.xxx)
    # This allows scripts to use model_config.xxx format
    if hasattr(cfg, 'model_config'):
        # Convert nested config to flat config
        cfg_dict = OmegaConf.to_container(cfg.model_config, resolve=True)
        # Merge with top-level config (top-level takes precedence)
        base_dict = OmegaConf.to_container(cfg, resolve=True)
        # Remove model_config from base_dict
        base_dict.pop('model_config', None)
        # Merge: base_dict overrides model_config values
        merged_dict = {**cfg_dict, **base_dict}
        cfg = OmegaConf.create(merged_dict)
    
    # Convert OmegaConf to dict for backward compatibility
    modelConfig = OmegaConf.to_container(cfg, resolve=True)
    
    # Convert None/null strings and boolean strings to actual types
    for key, value in modelConfig.items():
        if isinstance(value, str):
            if value.lower() == "none" or value.lower() == "null":
                modelConfig[key] = None
            elif value.lower() == "true":
                modelConfig[key] = True
            elif value.lower() == "false":
                modelConfig[key] = False
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    for key, value in sorted(modelConfig.items()):
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Run training or evaluation
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        from Diffusion.Train import eval as eval_func
        eval_func(modelConfig)


if __name__ == '__main__':
    # Set CUDA_VISIBLE_DEVICES if not already set
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8,9'
    
    main()
