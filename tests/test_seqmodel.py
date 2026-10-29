import os
import sys

import pytest
import torch

sys.path.append(os.getcwd())

from ldm.modules.seqmodel import CausalSliceTransformer, LatentDiffusionSchedule


def test_schedule_buffers_monotonic_and_bounded():
    sch = LatentDiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    acp = sch.sqrt_alphas_cumprod ** 2
    assert acp.shape == (1000,)
    assert torch.all(acp[1:] < acp[:-1])          # 累积乘积单调递减
    assert (acp > 0).all() and (acp < 1).all()


def test_q_sample_endpoints():
    sch = LatentDiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    x = torch.zeros(2, 4, 8, 8)
    noise = torch.ones(2, 4, 8, 8)
    t0 = torch.zeros(2, dtype=torch.long)
    tT = torch.full((2,), 999, dtype=torch.long)
    y0 = sch.q_sample(x, t0, noise)
    yT = sch.q_sample(x, tT, noise)
    # t=0：几乎还是 x 本身（x=0 → 幅度极小）
    assert y0.abs().mean() < 0.05
    # t=999：几乎纯噪声（幅度≈1）
    assert (yT.abs().mean() - 1.0).abs() < 0.05


def test_q_sample_batched_t():
    sch = LatentDiffusionSchedule(timesteps=1000)
    x = torch.randn(3, 4, 8, 8)
    noise = torch.randn_like(x)
    t = torch.tensor([0, 500, 999])
    y = sch.q_sample(x, t, noise)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def _tiny_seqmodel(latent_size=8):
    return CausalSliceTransformer(
        latent_channels=4, latent_size=latent_size, d_model=32,
        num_layers=2, num_heads=4, dim_feedforward=64, dropout=0.0)


def test_seqmodel_forward_shape():
    """专利 S3 修订：逐位置噪声预测，输出 (B, L, C, H, W)"""
    m = _tiny_seqmodel()
    x = torch.randn(2, 5, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    eps = m(x, t)
    assert eps.shape == (2, 5, 4, 8, 8)
    assert torch.isfinite(eps).all()


def test_causal_future_isolation():
    """篡改未来切片（含目标位）不得改变过去位置的隐状态（专利：因果掩码）"""
    torch.manual_seed(0)
    m = _tiny_seqmodel().eval()
    x = torch.randn(2, 5, 4, 8, 8)
    x2 = x.clone()
    x2[:, 3:] = torch.randn_like(x2[:, 3:])
    with torch.no_grad():
        f1 = m._features(x)
        f2 = m._features(x2)
    assert torch.allclose(f1[:, :3], f2[:, :3], atol=1e-5)
    assert not torch.allclose(f1[:, 3], f2[:, 3])


def test_seqmodel_timestep_changes_output():
    m = _tiny_seqmodel().eval()
    x = torch.randn(1, 5, 4, 8, 8)
    with torch.no_grad():
        eps_a = m(x, torch.zeros(1, dtype=torch.long))
        eps_b = m(x, torch.full((1,), 500, dtype=torch.long))
    assert not torch.allclose(eps_a, eps_b)


def test_extract_semantic_shape():
    m = _tiny_seqmodel()
    x = torch.randn(2, 5, 4, 8, 8)
    feats = m.extract_semantic(x)
    assert feats.shape == (2, 32, 8, 8)   # d_model=32


def test_seq_len_exceeds_max_raises():
    m = _tiny_seqmodel()
    x = torch.randn(1, 17, 4, 8, 8)       # max_seq_len 默认 16
    with pytest.raises(ValueError):
        m(x, None)
