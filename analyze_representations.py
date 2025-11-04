#!/usr/bin/env python3
"""
表征分析脚本
用于加载、分析和可视化训练过程中提取的模型表征
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from typing import List, Dict, Any


def load_representations(representation_dir: str, epoch: int = None) -> List[Dict[str, Any]]:
    """
    加载指定epoch的表征数据
    
    Args:
        representation_dir: 表征保存目录
        epoch: 指定epoch，如果为None则加载所有epoch
    
    Returns:
        表征数据列表
    """
    all_representations = []
    
    if epoch is not None:
        # 加载指定epoch的表征
        file_path = os.path.join(representation_dir, f'epoch_{epoch}_representations.pt')
        if os.path.exists(file_path):
            representations = torch.load(file_path, map_location='cpu')
            all_representations.extend(representations)
    else:
        # 加载所有epoch的表征
        for file_name in os.listdir(representation_dir):
            if file_name.startswith('epoch_') and file_name.endswith('_representations.pt'):
                file_path = os.path.join(representation_dir, file_name)
                representations = torch.load(file_path, map_location='cpu')
                all_representations.extend(representations)
    
    return all_representations


def analyze_representation_statistics(representations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析表征的统计特性
    
    Args:
        representations: 表征数据列表
    
    Returns:
        统计信息字典
    """
    if not representations:
        return {}
    
    # 收集所有表征
    all_reprs = torch.cat([r['representation'] for r in representations], dim=0)
    all_labels = torch.cat([r['labels'] for r in representations], dim=0)
    
    stats = {
        'total_samples': len(representations),
        'total_representations': all_reprs.shape[0],
        'representation_shape': all_reprs.shape,
        'mean': all_reprs.mean().item(),
        'std': all_reprs.std().item(),
        'min': all_reprs.min().item(),
        'max': all_reprs.max().item(),
        'label_distribution': torch.bincount(all_labels.flatten()).tolist(),
        'epochs': list(set([r['epoch'] for r in representations]))
    }
    
    return stats


def visualize_representations_tsne(representations: List[Dict[str, Any]], 
                                 save_path: str = None, 
                                 max_samples: int = 1000):
    """
    使用t-SNE可视化表征
    
    Args:
        representations: 表征数据列表
        save_path: 保存图片的路径
        max_samples: 最大样本数（用于加速计算）
    """
    if not representations:
        print("No representations to visualize")
        return
    
    # 收集表征和标签
    all_reprs = torch.cat([r['representation'] for r in representations], dim=0)
    all_labels = torch.cat([r['labels'] for r in representations], dim=0)
    
    # 如果样本太多，随机采样
    if all_reprs.shape[0] > max_samples:
        indices = torch.randperm(all_reprs.shape[0])[:max_samples]
        all_reprs = all_reprs[indices]
        all_labels = all_labels[indices]
    
    # 将表征展平为2D
    reprs_flat = all_reprs.view(all_reprs.shape[0], -1).numpy()
    labels_flat = all_labels.flatten().numpy()
    
    # 使用PCA降维（如果维度太高）
    if reprs_flat.shape[1] > 50:
        pca = PCA(n_components=50)
        reprs_flat = pca.fit_transform(reprs_flat)
        print(f"Applied PCA, explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 使用t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reprs_2d = tsne.fit_transform(reprs_flat)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reprs_2d[:, 0], reprs_2d[:, 1], c=labels_flat, 
                         cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class Label')
    plt.title('t-SNE Visualization of Model Representations')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()


def visualize_representation_evolution(representations: List[Dict[str, Any]], 
                                     save_path: str = None):
    """
    可视化表征随训练epoch的演化
    
    Args:
        representations: 表征数据列表
        save_path: 保存图片的路径
    """
    if not representations:
        print("No representations to visualize")
        return
    
    # 按epoch分组
    epoch_stats = {}
    for r in representations:
        epoch = r['epoch']
        if epoch not in epoch_stats:
            epoch_stats[epoch] = []
        epoch_stats[epoch].append(r['representation'].mean().item())
    
    epochs = sorted(epoch_stats.keys())
    means = [np.mean(epoch_stats[epoch]) for epoch in epochs]
    stds = [np.std(epoch_stats[epoch]) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(epochs, means, yerr=stds, marker='o', capsize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Representation Value')
    plt.title('Representation Evolution During Training')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze model representations')
    parser.add_argument('--representation_dir', type=str, required=True,
                       help='Directory containing representation files')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to analyze (default: all epochs)')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Output directory for analysis results')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples for t-SNE visualization')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载表征
    print(f"Loading representations from {args.representation_dir}")
    representations = load_representations(args.representation_dir, args.epoch)
    
    if not representations:
        print("No representations found!")
        return
    
    print(f"Loaded {len(representations)} representation batches")
    
    # 分析统计特性
    print("\nAnalyzing representation statistics...")
    stats = analyze_representation_statistics(representations)
    
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total representations: {stats['total_representations']}")
    print(f"Representation shape: {stats['representation_shape']}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Epochs: {stats['epochs']}")
    print(f"Label distribution: {stats['label_distribution']}")
    
    # 保存统计信息
    stats_path = os.path.join(args.output_dir, 'representation_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Representation Analysis Statistics\n")
        f.write("=" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # 生成可视化
    print("\nGenerating visualizations...")
    
    # t-SNE可视化
    tsne_path = os.path.join(args.output_dir, 'tsne_visualization.png')
    visualize_representations_tsne(representations, tsne_path, args.max_samples)
    
    # 表征演化可视化
    evolution_path = os.path.join(args.output_dir, 'representation_evolution.png')
    visualize_representation_evolution(representations, evolution_path)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
