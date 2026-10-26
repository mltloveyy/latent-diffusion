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
        """x_start: (B, C, H, W) 或 (B, L, C, H, W)，t: (B,) long → 加噪状态（与 x_start 同形）
        4-D 时 t 逐样本、5-D 时 t 逐样本且窗口内各切片共享同一时间步（专利 S3 修订）"""
        s1 = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))
        return s1 * x_start + s2 * noise


def timestep_embedding(t, dim, max_period=10000):
    """正弦时间步嵌入（同 DDPM）。t: (B,) → (B, dim)"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CausalSliceTransformer(nn.Module):
    """
    序列建模模块（SeqModel）。连续切片空间对齐，采用分解式注意力：
    每个空间位置 (h,w) 沿切片序列维度做因果自注意力（专利 S3：位置只聚合前序切片），
    末位置再经空间卷积块补充跨空间信息交换。
    """

    def __init__(self, latent_channels=4, latent_size=32, d_model=256, num_layers=4,
                 num_heads=8, dim_feedforward=1024, dropout=0.1, max_seq_len=16):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Conv2d(latent_channels, d_model, 1)
        self.pos_spatial = nn.Parameter(torch.randn(1, d_model, latent_size, latent_size) * 0.02)
        self.pos_temporal = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False)

        self.spatial_refine = nn.Sequential(
            nn.GroupNorm(8, d_model),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )
        self.eps_head = nn.Conv2d(d_model, latent_channels, 1)

    def _features(self, x):
        """x: (B, L, C, H, W) → 全位置隐状态 (B, L, D, H, W)（不含时间步注入与空间细化）"""
        B, L, C, H, W = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"序列长度 {L} 超过 max_seq_len={self.max_seq_len}")
        h = self.input_proj(x.reshape(B * L, C, H, W))            # (B*L, D, H, W)
        h = h + self.pos_spatial
        h = h.reshape(B, L, -1, H, W)
        # (B, L, D, H, W) → (B, H, W, L, D) → (B*H*W, L, D)：每个空间位置沿切片维度做因果注意力
        h = h.permute(0, 3, 4, 1, 2).reshape(B * H * W, L, -1)
        h = h + self.pos_temporal[:, :L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        h = self.transformer(h, mask=mask)
        # (B*H*W, L, D) → (B, H, W, L, D) → (B, L, D, H, W)
        h = h.reshape(B, H, W, L, -1).permute(0, 3, 4, 1, 2)
        return h

    def forward(self, x, t=None):
        """x: (B, L, C, H, W) 窗口内各切片按同一时间步 t 加噪后的潜变量序列；
        t: (B,) 扩散时间步（None 视为 0），作为全窗口共享条件。
        返回逐位置噪声预测 ε̂: (B, L, C, H, W)——因果掩码下位置 i 的预测
        仅依赖窗口起点至位置 i（含）的加噪潜变量与时间步 t（专利 S3 修订）。"""
        B, L, C, H, W = x.shape
        if t is None:
            t = torch.zeros(B, dtype=torch.long, device=x.device)
        h = self._features(x)                                       # (B, L, D, H, W)
        t_emb = self.time_embed(timestep_embedding(t, h.shape[2]))  # (B, D)
        h = h + t_emb[:, None, :, None, None]                       # 时间步注入全部位置
        feats = self.spatial_refine(h.reshape(B * L, -1, H, W))     # 逐位置空间细化
        return self.eps_head(feats).reshape(B, L, C, H, W)          # (B, L, C, H, W)

    def extract_semantic(self, x):
        """语义流特征（S4，专利修订）：输入前 k 张切片的干净潜变量（不含当前切片），
        等价于最小扩散时间步（t=0）的窗口形态；返回末位置隐状态经空间细化的
        (B, d_model, H, W)——蕴含前序切片沿序列方向演化至当前切片的预测性表征。"""
        B = x.shape[0]
        t0 = torch.zeros(B, dtype=torch.long, device=x.device)
        h = self._features(x)
        t_emb = self.time_embed(timestep_embedding(t0, h.shape[2]))
        h = h + t_emb[:, None, :, None, None]
        return self.spatial_refine(h[:, -1])                        # (B, d_model, H, W)
