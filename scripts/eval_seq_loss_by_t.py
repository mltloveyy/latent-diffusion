"""
按扩散时间步 t 分桶评估 S3 的 ε-预测损失（在验证体上）。

把训练日志里的平均 loss 拆成物理区，作为收敛/质量的最终判据：
  - t < ~100  : ε 藏在 ~1% 扰动里，即使上下文完美也几乎不可读 → 该段 ≈1（不可约）
  - t > ~600  : 输入本身就是 ε（√(1-ᾱ)→1）→ 该段 ≈0（平凡可读）
  - 中间过渡带: ε̂* = E[ε|z^t, ctx] —— 该段的平台值才是"上下文推断能力"与收敛的终判
固定 t 列表与噪声种子后结果可复现，可用于跨 checkpoint 对比。

用法（仓库根目录）:
    python scripts/eval_seq_loss_by_t.py \
        --base configs/seqmodel/latent_seq_fp.yaml \
        --ckpt logs/<run>/checkpoints/best-epoch=0024-step=009549.ckpt
可选: --timesteps 10 50 100 200 300 500 800  --max_windows 400 --batch_size 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/seqmodel/latent_seq_fp.yaml", help="S3 训练 yaml")
    ap.add_argument("--ckpt", required=True, help="S3 checkpoint（Lightning 格式）")
    ap.add_argument("--timesteps", type=int, nargs="+",
                    default=[10, 50, 100, 200, 300, 500, 800], help="评估的固定 t 列表")
    ap.add_argument("--max_windows", type=int, default=400, help="参与评估的窗口数上限（均匀抽取）")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = instantiate_from_config(OmegaConf.load(args.base).model)
    model.init_from_ckpt(args.ckpt)
    model.eval()

    # 验证集（与训练同源配置：val_volumes / stride / 预处理）
    ds_cfg = OmegaConf.load(args.base).data.params.validation
    ds = instantiate_from_config(ds_cfg)
    n = len(ds)
    if n == 0:
        raise RuntimeError("验证集为空——检查 yaml 的 val_volumes")
    if n > args.max_windows:
        sel = np.linspace(0, n - 1, args.max_windows).round().astype(int)
    else:
        sel = np.arange(n)
    print(f"验证窗口: {len(sel)}/{n}  (val_volumes={ds.split_volumes})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    t_max = model.schedule.timesteps
    skipped = [t for t in args.timesteps if t > t_max - 1]
    if skipped:
        print(f"注意: {skipped} 超出调度范围 (timesteps={t_max})，已截断到 {t_max - 1}")
    t_list = sorted({min(t, t_max - 1) for t in args.timesteps})

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    sums = {t: 0.0 for t in t_list}
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(sel), args.batch_size):
            batch_idx = sel[i:i + args.batch_size]
            seqs = np.stack([ds[int(j)]["seq"] for j in batch_idx])
            # 绕过了 Lightning，需手动把 batch 搬到模型所在设备
            batch = {"seq": torch.from_numpy(seqs).to(device)}
            z_seq = model.get_latents(batch).to(device)          # (b, L, C, h, w) 已缩放
            for t_val in t_list:
                t = torch.full((z_seq.shape[0],), t_val, dtype=torch.long, device=device)
                noise = torch.randn(z_seq.shape, generator=gen).to(device)
                z_noisy = model.schedule.q_sample(z_seq, t, noise)
                eps_pred = model.model(z_noisy, t)
                mse = torch.mean((eps_pred - noise) ** 2, dim=(1, 2, 3, 4))  # 每窗口
                sums[t_val] += mse.sum().item()
            cnt += len(batch_idx)

    print(f"\n{'t':>6}  {'MSE':>10}   物理区")
    bands = {10: "不可约区", 50: "不可约区", 100: "过渡带", 200: "过渡带",
             300: "过渡带", 500: "平凡可读区", 800: "平凡可读区"}
    avg = 0.0
    for t_val in t_list:
        m = sums[t_val] / cnt
        avg += m / len(t_list)
        print(f"{t_val:>6}  {m:>10.4f}   {bands.get(t_val, '')}")
    print(f"\n均匀 t 平均 = {avg:.4f}（对比训练日志的 val/loss_simple≈{0.12}，此处 t 采样不同略有差异）")
    print("判读: 不可约区≈1、平凡区≈0 属结构性；过渡带平台值≈上下文推断能力的终判。")


if __name__ == "__main__":
    main()
