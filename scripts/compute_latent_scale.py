"""
计算 VAE 潜变量标准差，作为 S3 扩散的 scale_factor（同 LDM 惯例，避免 kl_weight=1e-6
导致的潜变量幅度漂移使加噪调度失真）。

用法（在仓库根目录，VAE 权重同步回来后执行）:
    python scripts/compute_latent_scale.py \
        --base configs/autoencoder/autoencoder_kl_32x32x4_fp.yaml \
        --ckpt <vae.ckpt路径> --num_slices 256
把打印出的 scale_factor 填入 configs/seqmodel/latent_seq_fp.yaml。
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="AE 训练 yaml")
    parser.add_argument("--ckpt", required=True, help="AE checkpoint 路径")
    parser.add_argument("--data_root", default="data/fingerprint/UESTC")
    parser.add_argument("--num_slices", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    opt = parser.parse_args()

    config = OmegaConf.load(opt.base)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(opt.ckpt)
    model.eval()

    filelist_path = os.path.join(opt.data_root, "dataset", "filelist.txt")
    with open(filelist_path) as f:
        filelist = f.read().splitlines()
    random.seed(opt.seed)
    paths = random.sample(filelist, min(opt.num_slices, len(filelist)))

    import albumentations
    from PIL import Image

    pre = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=256),
        albumentations.CenterCrop(height=256, width=256),
    ])

    def load(p):
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = pre(image=np.array(img).astype(np.uint8))["image"]
        return (img / 127.5 - 1.0).astype(np.float32)

    vals = []
    with torch.no_grad():
        for i in range(0, len(paths), opt.batch_size):
            x = torch.from_numpy(np.stack([load(p) for p in paths[i:i + opt.batch_size]]))
            x = x.permute(0, 3, 1, 2)
            z = model.encode(x).mode()
            vals.append(z.flatten())
    std = torch.cat(vals).std().item()
    print(f"潜变量标准差 = {std:.6f}")
    print(f"建议在 configs/seqmodel/latent_seq_fp.yaml 中设置 scale_factor: {1.0 / std:.6f}")
    print("（LDM 惯例：encode 乘 scale_factor / decode 除回，scale_factor = 1/std，使扩散在 std≈1 的潜空间进行）")


if __name__ == "__main__":
    main()
