"""
基于潜在扩散的三维上下文建模（专利 S3）：
冻结 VAE 编码连续切片序列 → 上下文潜变量保持干净、目标切片潜变量加噪
→ 因果 Transformer 预测 ε（简化噪声预测损失，无监督）。
"""
import torch
import torch.nn.functional as F

import lightning.pytorch as pl

from ldm.modules.seqmodel import LatentDiffusionSchedule
from ldm.util import instantiate_from_config


class LatentSeqDiffusion(pl.LightningModule):
    def __init__(self,
                 seqmodel_config,
                 first_stage_config,
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=2e-2,
                 scale_factor=1.0,
                 ckpt_path=None,
                 ignore_keys=list(),
                 monitor="val/loss_simple",
                 ):
        super().__init__()
        self.monitor = monitor
        self.timesteps = timesteps
        self.scale_factor = scale_factor

        # 冻结的 VAE（ckpt_path 为空字符串视为未提供，等权重同步后再填）
        first_stage_config = dict(first_stage_config)
        first_stage_config["params"] = dict(first_stage_config.get("params", {}))
        ckpt = first_stage_config["params"].get("ckpt_path")
        first_stage_config["params"]["ckpt_path"] = ckpt if ckpt else None
        first_stage_config["params"]["lossconfig"] = {"target": "torch.nn.Identity"}
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.first_stage_model.eval()
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

        self.model = instantiate_from_config(seqmodel_config)
        self.schedule = LatentDiffusionSchedule(timesteps, beta_start, beta_end)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def on_train_epoch_start(self):
        self.first_stage_model.eval()   # 冻结模块防止进入训练态

    @torch.no_grad()
    def encode_first_stage(self, x):
        """x: (N, 3, H, W) → 后验均值潜变量 (N, C, h, w)"""
        posterior = self.first_stage_model.encode(x)
        return posterior.mode()

    @torch.no_grad()
    def decode_first_stage(self, z):
        """z: (B, C, h, w)（已缩放）→ 图像 (B, 3, H, W)"""
        return self.first_stage_model.decode(z / self.scale_factor)

    def get_latents(self, batch):
        """batch["seq"]: (B, L, H, W, 3) → (B, L, C, h, w)，已乘 scale_factor"""
        seq = batch["seq"]
        B, L = seq.shape[:2]
        x = seq.to(memory_format=torch.contiguous_format).float()
        x = x.permute(0, 1, 4, 2, 3)                    # (B, L, C, H, W)
        x = x.reshape(B * L, x.shape[2], x.shape[3], x.shape[4])
        z = self.encode_first_stage(x)                  # (B*L, C, h, w)
        z = z.reshape(B, L, *z.shape[1:])
        return z * self.scale_factor

    def forward(self, x_seq, t):
        """x_seq: (B, L, C, h, w) 已缩放；t: (B,) long → (ε̂, noise)"""
        B = x_seq.shape[0]
        noise = torch.randn(B, *x_seq.shape[2:], device=x_seq.device, dtype=x_seq.dtype)
        z_noisy = self.schedule.q_sample(x_seq[:, -1], t, noise)
        z_in = torch.cat([x_seq[:, :-1], z_noisy.unsqueeze(1)], dim=1)
        eps_pred = self.model(z_in, t)
        return eps_pred, noise

    def shared_step(self, batch, split):
        z_seq = self.get_latents(batch)
        B = z_seq.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z_seq.device, dtype=torch.long)
        eps_pred, noise = self(z_seq, t)
        loss = F.mse_loss(eps_pred, noise)
        self.log(f"{split}/loss_simple", loss,
                 prog_bar=True, logger=True,
                 on_step=(split == "train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")   # 键名 val/loss_simple 与 monitor 及仓库惯例一致

    def extract_semantic(self, z_seq):
        """S4 语义流接口：潜变量序列（已缩放）→ 末位置语义特征 (B, d_model, h, w)"""
        return self.model.extract_semantic(z_seq)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),
                                 lr=self.learning_rate, weight_decay=0.01)

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        z_seq = self.get_latents(batch)
        B = z_seq.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z_seq.device, dtype=torch.long)
        eps_pred, noise = self(z_seq, t)
        z_noisy = self.schedule.q_sample(z_seq[:, -1], t, noise)
        s1 = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_hat = (z_noisy - s2 * eps_pred) / s1
        return {
            "context": self.decode_first_stage(z_seq[:, -2]),
            "target": self.decode_first_stage(z_seq[:, -1]),
            "denoise_x0": self.decode_first_stage(x0_hat),
        }
