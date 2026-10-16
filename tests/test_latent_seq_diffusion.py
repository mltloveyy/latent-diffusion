import os
import sys

import torch

sys.path.append(os.getcwd())

from ldm.models.seq_diffusion import LatentSeqDiffusion

TINY_DD = dict(double_z=True, z_channels=4, resolution=64, in_channels=3, out_ch=3,
               ch=32, ch_mult=[1, 2, 4], num_res_blocks=1, attn_resolutions=[], dropout=0.0)
# ch_mult 3 级 → 2 次下采样：64→32→16，f=4 → 潜空间 16×16×4
# ch 必须被 32 整除（CompVis Encoder 的 ResnetBlock 固定 GroupNorm(32, ch)）


def _tiny_model(**overrides):
    kwargs = dict(
        seqmodel_config={"target": "ldm.modules.seqmodel.CausalSliceTransformer",
                         "params": dict(latent_channels=4, latent_size=16, d_model=32,
                                        num_layers=1, num_heads=4, dim_feedforward=64)},
        first_stage_config={"target": "ldm.models.autoencoder.AutoencoderKL",
                            "params": dict(ddconfig=TINY_DD, embed_dim=4, ckpt_path="")},
        timesteps=20,
    )
    kwargs.update(overrides)
    return LatentSeqDiffusion(**kwargs)


def _fake_batch(B=1, L=3, size=64):
    return {"seq": torch.rand(B, L, size, size, 3) * 2 - 1}


def test_ckpt_path_empty_string_is_none():
    """VAE ckpt 未同步时 yaml 里填空字符串，不得触发 init_from_ckpt 崩溃"""
    m = _tiny_model()
    assert m.first_stage_model is not None   # 构造成功即通过


def test_vae_frozen_and_eval():
    m = _tiny_model()
    assert all(not p.requires_grad for p in m.first_stage_model.parameters())
    assert not m.first_stage_model.training


def test_get_latents_scale_factor():
    m = _tiny_model()
    batch = _fake_batch()
    z1 = m.get_latents(batch)
    m.scale_factor = 2.0
    z2 = m.get_latents(batch)
    assert torch.allclose(z2, z1 * 2.0, atol=1e-5)
    assert z1.shape == (1, 3, 4, 16, 16)


def test_training_step_finite():
    m = _tiny_model()
    loss = m.training_step(_fake_batch(), 0)
    assert torch.isfinite(loss)


def test_validation_step_no_crash():
    """validation_step 在未挂 trainer 时允许告警但不得抛异常"""
    m = _tiny_model()
    out = m.validation_step(_fake_batch(), 0)
    assert torch.isfinite(out)


def test_monitor_attr():
    m = _tiny_model()
    assert m.monitor == "val/loss_simple"


def test_log_images_keys():
    m = _tiny_model()
    log = m.log_images(_fake_batch())
    assert set(log.keys()) == {"context", "target", "denoise_x0"}
    for v in log.values():
        assert v.shape == (1, 3, 64, 64)


def test_extract_semantic_passthrough():
    m = _tiny_model()
    z_seq = m.get_latents(_fake_batch())
    feats = m.extract_semantic(z_seq)
    assert feats.shape == (1, 32, 16, 16)
