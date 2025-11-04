from DiffusionFreeGuidence.TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 100,
        "batch_size": 256,
        "T": 3000,
        "channel": 128,
        "channel_mult": [1, 4, 8, 8, 4, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/ep_100_bsz256_T3000_lr5e-5_reprs",
        "training_load_weight": None,
        "test_load_weight": "/home/yxfeng/project2/DenoisingDiffusionProbabilityModel-ddpm-/CheckpointsCondition/ckpt_63_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "SampledGuidenceImgs1.png",
        "sampledImgName": "SampledGuidenceImgs2.png",
        "nrow": 8,
        # 表征提取配置
        "extract_representation_freq": 50,  # 每隔多少个batch提取一次表征，设为0表示不提取
        "save_representations": True,  # 是否保存表征
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
