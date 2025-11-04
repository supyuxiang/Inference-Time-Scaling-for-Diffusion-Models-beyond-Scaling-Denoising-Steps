import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Optional, Union, List, Tuple
import warnings

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, using torch-only implementation for sqrtm")

class FID:
    '''
    Frechet Inception Distance Calculator
    完整的FID计算器，包含特征提取和距离计算
    公式为：
    FID = ||μ_real - μ_fake||_2^2 + Tr(Σ_real + Σ_fake - 2(Σ_realΣ_fake)^1/2)
    其中，μ_real和μ_fake分别为真实数据和生成数据的均值，Σ_real和Σ_fake分别为真实数据和生成数据的协方差矩阵。
    '''
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.inception_model = self._load_inception_model()
        self.preprocess = self._get_preprocess_transform()
        
    def _load_inception_model(self):
        '''加载Inception v3模型用于特征提取'''
        # 使用新的 weights API（pretrained=True 已废弃）
        # 先不设置 aux_logits，加载后再修改
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        # 移除辅助输出分支
        inception.AuxLogits = None
        inception.aux_logits = False
        # 移除最后的分类层，直接输出 2048 维特征
        inception.fc = nn.Identity()
        inception = inception.to(self.device)
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        return inception
    
    def _get_preprocess_transform(self):
        '''获取图片预处理transform'''
        return transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(299),  # Inception v3 的标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features_from_tensor(self, images: torch.Tensor) -> torch.Tensor:
        '''
        从 tensor 格式的图像中提取 Inception 特征
        
        Args:
            images: 图像张量 (N, C, H, W)，值域应该在 [0, 1]
            
        Returns:
            features: 提取的特征张量 (N, 2048)
        '''
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor (N, C, H, W), got {images.dim()}D")
        
        # 调整到 299x299 并标准化
        images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 标准化到 ImageNet 的均值和方差
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std
        
        with torch.no_grad():
            batch_features = self.inception_model(images_normalized.to(self.device))
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]
        
        return batch_features
    
    def extract_features(self, image_paths: List[str], batch_size: int = 32):
        '''
        从图片路径列表中提取特征
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
            
        Returns:
            features: 提取的特征张量 (N, 2048)
        '''
        features = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                
                # 加载并预处理图片
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert('RGB')
                        image = self.preprocess(image)
                        batch_images.append(image)
                    except Exception as e:
                        print(f"Error loading image {path}: {e}")
                        continue
                
                if not batch_images:
                    continue
                    
                # 批量处理
                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_features = self.inception_model(batch_tensor)
                
                # 处理可能的 tuple 返回值（虽然 aux_logits=False，但为了安全）
                if isinstance(batch_features, tuple):
                    batch_features = batch_features[0]
                
                features.append(batch_features.cpu())
        
        if not features:
            raise ValueError("No features were extracted from the images")
            
        return torch.cat(features, dim=0)
    
    def calculate_frechet_distance(self, mu1: torch.Tensor, sigma1: torch.Tensor, 
                                  mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6):
        '''
        计算两个多元高斯分布之间的Fréchet距离
        
        Args:
            mu1, sigma1: 第一个分布的均值和协方差
            mu2, sigma2: 第二个分布的均值和协方差
            eps: 数值稳定性参数
            
        Returns:
            fid: Fréchet距离
        
        公式为：
        FID = ||μ_real - μ_fake||_2^2 + Tr(Σ_real + Σ_fake - 2(Σ_realΣ_fake)^(1/2))
        '''
        mu1 = mu1.to(torch.float64)
        mu2 = mu2.to(torch.float64)
        sigma1 = sigma1.to(torch.float64)
        sigma2 = sigma2.to(torch.float64)
        
        # 计算均值之差的平方
        diff = mu1 - mu2
        diff_squared = torch.dot(diff, diff)
        
        # 计算协方差矩阵的迹部分
        # 使用更稳定的方法计算矩阵平方根
        # 添加小的正则化项以提高数值稳定性
        offset = torch.eye(sigma1.shape[0], dtype=torch.float64, device=sigma1.device) * eps
        covmean = self._sqrtm((sigma1 + offset) @ (sigma2 + offset))
        
        # 确保结果是实数（去除小的虚部）
        if torch.is_complex(covmean):
            covmean = covmean.real
        if torch.any(torch.isnan(covmean)):
            # 如果还有 NaN，使用更保守的近似
            covmean = torch.eye(sigma1.shape[0], dtype=torch.float64, device=sigma1.device) * torch.sqrt(
                torch.trace((sigma1 + offset) @ (sigma2 + offset)) / sigma1.shape[0]
            )
        
        trace_term = torch.trace(sigma1 + sigma2 - 2.0 * covmean)
        
        fid = (diff_squared + trace_term).clamp_min(0.0)  # 确保非负
        return fid.item()
    
    def _sqrtm(self, matrix: torch.Tensor):
        '''计算矩阵的平方根（使用特征分解方法）'''
        if SCIPY_AVAILABLE:
            # 使用 scipy 的 sqrtm（更稳定）
            matrix_np = matrix.cpu().double().numpy()
            try:
                sqrtm_matrix = linalg.sqrtm(matrix_np)
                if np.iscomplexobj(sqrtm_matrix):
                    sqrtm_matrix = sqrtm_matrix.real
                return torch.from_numpy(sqrtm_matrix).to(matrix.device)
            except:
                pass  # 回退到 torch 方法
        
        # 备用方法：特征分解（torch-only）
        matrix = matrix.double()
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        eigenvalues = eigenvalues.real.clamp_min(0.0)  # 确保非负
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        # 处理复数部分（虽然已经取 real，但为了类型一致）
        eigenvectors = eigenvectors.real
        sqrt_matrix = eigenvectors @ torch.diag(sqrt_eigenvalues) @ torch.linalg.pinv(eigenvectors)
        return sqrt_matrix.real
    
    def calculate_activation_statistics(self, features: torch.Tensor):
        '''
        计算特征集的统计量（均值和协方差）
        
        Args:
            features: 特征张量 (N, dim)
            
        Returns:
            mu: 均值向量
            sigma: 协方差矩阵
        '''
        if len(features) == 1:
            warnings.warn("Only one sample provided for statistics calculation. "
                         "Covariance matrix will be zero.")
        
        mu = torch.mean(features, dim=0)
        
        if len(features) > 1:
            # FID 标准实现使用有偏估计（除以 N，而不是 N-1）
            # features shape: (N, dim)，需要转置为 (dim, N) 给 torch.cov
            sigma = torch.cov(features.T, correction=0)  # correction=0 表示除以 N
        else:
            # 单样本时协方差为零，使用小的正则化项
            sigma = torch.zeros((features.shape[1], features.shape[1]), 
                               dtype=features.dtype, device=features.device)
            sigma = sigma + 1e-6 * torch.eye(features.shape[1], 
                                            dtype=features.dtype, device=features.device)
            
        return mu, sigma
    
    def compute_fid(self, real_features: torch.Tensor, fake_features: torch.Tensor):
        '''
        计算两个特征集之间的FID
        
        Args:
            real_features: 真实图片特征 (N, 2048)
            fake_features: 生成图片特征 (M, 2048)
            
        Returns:
            fid_score: FID分数
        '''
        mu_real, sigma_real = self.calculate_activation_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_activation_statistics(fake_features)
        
        fid_score = self.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid_score


class IS:
    '''
    Inception Score Calculator
    完整的 IS 计算器，包含特征提取和分数计算
    公式：IS = exp(E[KL(p(y|x)||p(y))])
    其中 p(y|x) 是每张图像的分类概率，p(y) 是边缘分布
    '''
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.inception_model = self._load_inception_model()
        self.preprocess = self._get_preprocess_transform()
    
    def _load_inception_model(self):
        '''加载完整的 Inception v3 模型（保留分类层用于 IS）'''
        # 先不设置 aux_logits，加载后再修改
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        # 移除辅助输出分支
        inception.AuxLogits = None
        inception.aux_logits = False
        inception = inception.to(self.device)
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        return inception
    
    def _get_preprocess_transform(self):
        '''获取图片预处理transform'''
        return transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_logits_from_tensor(self, images: torch.Tensor) -> torch.Tensor:
        '''
        从 tensor 格式的图像中提取 Inception 分类 logits
        
        Args:
            images: 图像张量 (N, C, H, W)，值域应该在 [0, 1]
            
        Returns:
            logits: 分类 logits (N, 1000)
        '''
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor (N, C, H, W), got {images.dim()}D")
        
        # 调整到 299x299 并标准化
        images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 标准化到 ImageNet 的均值和方差
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std
        
        with torch.no_grad():
            logits = self.inception_model(images_normalized.to(self.device))
            if isinstance(logits, tuple):
                logits = logits[0]
        
        return logits
    
    def compute_is(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        '''
        计算 Inception Score
        
        Args:
            images: 图像张量 (N, C, H, W)，值域 [0, 1]
            splits: 用于计算 IS 的分割数（用于估计方差）
            
        Returns:
            mean_is: IS 的均值
            std_is: IS 的标准差
        '''
        # 提取 logits
        logits = self.extract_logits_from_tensor(images)  # (N, 1000)
        
        # 计算分类概率
        probs = F.softmax(logits, dim=1)  # (N, 1000)
        
        # 计算边缘分布
        p_y = probs.mean(dim=0, keepdim=True)  # (1, 1000)
        p_y = p_y.clamp_min(1e-12)
        
        # 计算每张图像的 KL 散度
        kl = (probs * (torch.log(probs.clamp_min(1e-12)) - torch.log(p_y))).sum(dim=1)  # (N,)
        is_scores = torch.exp(kl)  # (N,)
        
        # 使用 splits 来估计均值和标准差（标准做法）
        if len(is_scores) < splits:
            mean_is = is_scores.mean().item()
            std_is = is_scores.std(unbiased=False).item()
        else:
            split_scores = []
            chunk_size = len(is_scores) // splits
            for i in range(splits):
                start = i * chunk_size
                end = start + chunk_size if i < splits - 1 else len(is_scores)
                split_scores.append(is_scores[start:end].mean().item())
            mean_is = np.mean(split_scores)
            std_is = np.std(split_scores)
        
        return float(mean_is), float(std_is)


class CLIPScore:
    '''
    CLIP Score Calculator
    完整的 CLIP Score 计算器，用于评估生成图像的质量
    公式：CLIPScore = cosine_similarity(f(real), f(fake))
    这里我们计算生成图像与真实图像之间的 CLIP 特征相似度
    '''
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", model_name="ViT-B/32"):
        self.device = device
        self.model_name = model_name
        self.clip_model = self._load_clip_model()
        self.preprocess = self._get_preprocess_transform()
    
    def _load_clip_model(self):
        '''加载 CLIP 模型'''
        try:
            import clip
            model, _ = clip.load(self.model_name, device=self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Please install it with: pip install git+https://github.com/openai/CLIP.git"
            )
    
    def _get_preprocess_transform(self):
        '''获取 CLIP 预处理transform'''
        try:
            import clip
            return clip.load(self.model_name, device=self.device)[1]
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Please install it with: pip install git+https://github.com/openai/CLIP.git"
            )
    
    def extract_features_from_tensor(self, images: torch.Tensor) -> torch.Tensor:
        '''
        从 tensor 格式的图像中提取 CLIP 特征（优化版本：批量处理）
        
        Args:
            images: 图像张量 (N, C, H, W)，值域应该在 [0, 1]
            
        Returns:
            features: 提取的 CLIP 特征 (N, feature_dim)
        '''
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor (N, C, H, W), got {images.dim()}D")
        
        # CLIP 模型期望输入范围是 [0, 1]
        # 调整图像尺寸到 CLIP 的输入尺寸（通常是 224x224）
        images_resized = F.interpolate(images, size=224, mode='bilinear', align_corners=False)
        
        # 使用 CLIP 的预处理
        with torch.no_grad():
            # 将 [0, 1] 范围转换为 PIL Image，然后再用 CLIP 的 transform
            # 但为了效率，我们直接使用 tensor 计算
            # CLIP 的 transform 会进行标准化，我们需要手动处理
            # 简单方法：直接使用图像（CLIP 可以接受 tensor）
            # 但我们使用预处理 transform
            batch_size = images.shape[0]
            processed_images = []
            for i in range(batch_size):
                # 转换为 PIL Image 格式
                img = images_resized[i].cpu().clamp(0, 1)
                img = transforms.ToPILImage()(img)
                processed = self.preprocess(img)
                processed_images.append(processed)
            
            processed_tensor = torch.stack(processed_images).to(self.device)
            
            # 提取特征
            image_features = self.clip_model.encode_image(processed_tensor)
            # L2 normalize
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_features
    
    def compute_clip_score(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor,
        mode: str = "mean_similarity"
    ) -> float:
        '''
        计算 CLIP Score
        
        Args:
            real_images: 真实图像 (N, C, H, W)，值域 [0, 1]
            fake_images: 生成图像 (M, C, H, W)，值域 [0, 1]
            mode: 计算模式
                - "mean_similarity": 计算平均相似度（生成图像与真实图像的平均相似度）
                - "diversity": 计算生成图像之间的多样性（较低的相似度表示更高的多样性）
        
        Returns:
            clip_score: CLIP 分数
        '''
        if mode == "mean_similarity":
            # 提取特征
            real_features = self.extract_features_from_tensor(real_images)  # (N, dim)
            fake_features = self.extract_features_from_tensor(fake_images)  # (M, dim)
            
            # 计算平均相似度
            # 方法：计算每个生成图像与所有真实图像的平均相似度，然后取平均
            similarities = torch.matmul(fake_features, real_features.T)  # (M, N)
            clip_score = similarities.mean().item()
            
        elif mode == "diversity":
            # 计算生成图像之间的多样性（相似度越低，多样性越高）
            fake_features = self.extract_features_from_tensor(fake_images)  # (M, dim)
            
            if fake_features.shape[0] < 2:
                return 0.0
            
            # 计算所有生成图像对之间的相似度
            similarities = torch.matmul(fake_features, fake_features.T)  # (M, M)
            # 移除对角线（自身相似度为1）
            mask = ~torch.eye(fake_features.shape[0], dtype=torch.bool, device=self.device)
            pairwise_sim = similarities[mask]
            # 多样性 = 1 - 平均相似度（相似度越低，多样性越高）
            clip_score = (1.0 - pairwise_sim.mean().item())
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return float(clip_score)
    
    def compute_clip_score_with_features(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor
    ) -> float:
        '''
        使用已提取的特征计算 CLIP Score
        
        Args:
            real_features: 真实图像特征 (N, dim)，已 L2 归一化
            fake_features: 生成图像特征 (M, dim)，已 L2 归一化
        
        Returns:
            clip_score: CLIP 分数（平均相似度）
        '''
        similarities = torch.matmul(fake_features, real_features.T)  # (M, N)
        clip_score = similarities.mean().item()
        return float(clip_score)


if __name__ == "__main__":
    pass
























