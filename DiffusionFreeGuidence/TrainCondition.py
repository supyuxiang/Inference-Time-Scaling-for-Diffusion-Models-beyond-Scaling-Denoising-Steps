

import os
from typing import Dict
import numpy as np
import time

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    # 创建表征保存目录
    representation_dir = os.path.join(modelConfig["save_dir"], "representations")
    os.makedirs(representation_dir, exist_ok=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    start_time = time.time()
    for e in tqdm(range(modelConfig["epoch"]), desc="Training"):
        epoch_representations = []  # 存储当前epoch的表征
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch_idx, (images, labels) in enumerate(tqdmDataLoader):
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                
                # 检查是否需要提取表征（每隔一定步数或特定epoch）
                extract_representation = (modelConfig.get("extract_representation_freq", 0) > 0 and 
                                        batch_idx % modelConfig.get("extract_representation_freq", 100) == 0)
                
                if extract_representation:
                    loss, representation = trainer(x_0, labels, return_representation=True)
                    # 保存表征到CPU并存储
                    epoch_representations.append({
                        'epoch': e,
                        'batch_idx': batch_idx,
                        'representation': representation.detach().cpu(),
                        'labels': labels.detach().cpu(),
                        'images': x_0.detach().cpu()
                    })
                    print('========================================')
                    print(f"Extracted representation for batch {batch_idx}")
                    print(f'epoch_representations: {epoch_representations[-1].get("representation")}')
                    print(f'representation shape: {epoch_representations[-1].get("representation").shape}')
                    print('========================================')
                else:
                    loss = trainer(x_0, labels)
                
                loss = loss.sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    "reprs": len(epoch_representations)
                })
        
        # 保存当前epoch的表征
        if epoch_representations:
            representation_save_path = os.path.join(representation_dir, f'epoch_{e}_representations.pt')
            os.makedirs(os.path.dirname(representation_save_path), exist_ok=True)
            torch.save(epoch_representations, representation_save_path)
            print(f"Saved {len(epoch_representations)} representations for epoch {e}")
        
        warmUpScheduler.step()
        # 获取当前学习率（避免警告）
        current_lr = warmUpScheduler.get_last_lr()[0]
        save_path = os.path.join(modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(net_model.state_dict(), save_path)
        print(f"Epoch {e} completed, LR: {current_lr:.6f}, Loss: {loss.item():.6f}, Time: {time.time() - start_time:.2f}s")


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])