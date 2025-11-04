
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h



class PatchEmbedding(nn.Module):
    """将图像切分成patches并嵌入"""
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.initialize()
    
    def initialize(self):
        init.xavier_uniform_(self.proj.weight)
        init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # x: [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.proj(x)
        # Flatten: [B, embed_dim, H//patch_size, W//patch_size] -> [B, embed_dim, n_patches]
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x, temb=None):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # Add time embedding if provided
        if temb is not None:
            # temb: [B, tdim] -> [B, 1, embed_dim]
            temb = temb.unsqueeze(1)
            x = x + temb
        
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        
        return x


class ViT(nn.Module):
    """
    Vision Transformer for Diffusion Models
    
    Args:
        T: number of timesteps
        img_size: input image size (assumed to be square)
        patch_size: patch size (e.g., 4, 8, 16)
        in_chans: input channels (default: 3 for RGB)
        embed_dim: embedding dimension (default: 768)
        depth: number of transformer blocks (default: 12)
        num_heads: number of attention heads (default: 12)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        dropout: dropout rate (default: 0.1)
    """
    def __init__(self, T, img_size=256, patch_size=16, in_chans=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, 
                 ch=None, ch_mult=None, attn=None, num_res_blocks=None):
        super().__init__()
        # Note: ch, ch_mult, attn, num_res_blocks are kept for compatibility but not used
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Time embedding
        tdim = embed_dim * 4
        self.time_embedding = TimeEmbedding(T, embed_dim, tdim)
        # Project time embedding to match embed_dim
        self.temb_proj = nn.Linear(tdim, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head: reconstruct image from patches
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        
        self.initialize()
    
    def initialize(self):
        # Initialize position embedding
        init.normal_(self.pos_embed, std=0.02)
        
        # Initialize output head
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
    
    def forward(self, x, t):
        """
        Args:
            x: [B, C, H, W] input image
            t: [B] timestep indices
        Returns:
            [B, C, H, W] predicted noise
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Time embedding
        temb = self.time_embedding(t)  # [B, tdim]
        temb = self.temb_proj(temb)  # [B, embed_dim]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, temb)
        
        # Final norm
        x = self.norm(x)  # [B, n_patches, embed_dim]
        
        # Output head: predict patch values
        x = self.head(x)  # [B, n_patches, patch_size^2 * C]
        
        # Reshape to image
        patch_size = self.patch_size
        n_patches_per_side = H // patch_size
        x = x.reshape(B, n_patches_per_side, n_patches_per_side, patch_size, patch_size, C)
        # Rearrange: [B, H/p, W/p, p, p, C] -> [B, H, W, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x



if __name__ == '__main__':
    batch_size = 8
    
    # Test UNet
    print("Testing UNet...")
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(f"UNet output shape: {y.shape}")
    
    # Test ViT
    print("\nTesting ViT...")
    vit_model = ViT(
        T=1000, 
        img_size=256, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        dropout=0.1
    )
    x_vit = torch.randn(batch_size, 3, 256, 256)
    t_vit = torch.randint(1000, (batch_size, ))
    y_vit = vit_model(x_vit, t_vit)
    print(f"ViT output shape: {y_vit.shape}")
    
    # Test ViT with smaller image
    print("\nTesting ViT with 224x224 image...")
    vit_model_small = ViT(
        T=1000, 
        img_size=224, 
        patch_size=14, 
        embed_dim=384, 
        depth=6, 
        num_heads=6, 
        dropout=0.1
    )
    x_vit_small = torch.randn(batch_size, 3, 224, 224)
    y_vit_small = vit_model_small(x_vit_small, t_vit)
    print(f"ViT (small) output shape: {y_vit_small.shape}")

