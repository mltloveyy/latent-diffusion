import os
import sys

import pytest
import torch

sys.path.append(os.getcwd())

from ldm.modules.seqmodel import LatentDiffusionSchedule


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
