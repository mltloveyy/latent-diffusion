"""
潜空间扩散序列建模（专利 S3/S4 的 SeqModel 骨干）。

- LatentDiffusionSchedule: 线性 β 调度下的前向加噪 q_sample
- CausalSliceTransformer: 因果切片 Transformer，
  forward 输出目标切片的噪声预测 ε̂，extract_semantic 输出语义流特征（S4 用）
"""
import math

import torch
import torch.nn as nn


class LatentDiffusionSchedule(nn.Module):
    """线性方差调度：β 从 beta_start 线性增长至 beta_end（专利：T=1000, 1e-4→2e-2）"""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float(), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt().float(), persistent=False)
        self.timesteps = timesteps

    def q_sample(self, x_start, t, noise):
        """x_start: (B, C, H, W)，t: (B,) long → 加噪状态 (B, C, H, W)"""
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x_start + s2 * noise
