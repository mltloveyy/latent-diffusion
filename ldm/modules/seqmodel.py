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
        """x: (B, L, C, H, W)，末位为目标切片；t: (B,) 扩散时间步（None 视为 0）
        返回 ε̂: (B, C, H, W)"""
        B = x.shape[0]
        if t is None:
            t = torch.zeros(B, dtype=torch.long, device=x.device)
        h = self._features(x)
        t_emb = self.time_embed(timestep_embedding(t, h.shape[2]))  # (B, D)
        h = h.clone()
        h[:, -1] = h[:, -1] + t_emb[:, :, None, None]               # 时间步只注入目标位
        feats = self.spatial_refine(h[:, -1])                       # (B, D, H, W)
        return self.eps_head(feats)

    def extract_semantic(self, x):
        """语义流特征（S4）：末位置隐状态经空间细化，(B, d_model, H, W)。
        不注入时间步——语义特征与扩散时刻无关。"""
        h = self._features(x)
        return self.spatial_refine(h[:, -1])
