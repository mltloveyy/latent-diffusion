import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from ldm.modules.seqmodel import LatentDiffusionSchedule
from ldm.util import instantiate_from_config


class LatentSeqDiffusion(pl.LightningModule):
    def __init__(
        self,
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

        if ckpt_path:
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
        self.first_stage_model.eval()  # frozen VAE

    @torch.no_grad()
    def encode_first_stage(self, x):
        posterior = self.first_stage_model.encode(x)
        return posterior.mode()

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z / self.scale_factor)

    def get_latents(self, batch):
        seq = batch["seq"]  # (B, L, H, W, 3)
        B, L = seq.shape[:2]
        x = seq.to(memory_format=torch.contiguous_format).float()
        x = x.permute(0, 1, 4, 2, 3)  # (B, L, C, H, W)
        x = x.reshape(B * L, x.shape[2], x.shape[3], x.shape[4])  # (B*L, C, H, W)
        z = self.encode_first_stage(x)  # (B*L, C, h, w)
        z = z.reshape(B, L, *z.shape[1:])  # (B, L, C, h, w)
        return z * self.scale_factor

    def forward(self, x_seq, t):
        noise = torch.randn_like(x_seq)
        z_noisy = self.schedule.q_sample(x_seq, t, noise)
        eps_pred = self.model(z_noisy, t)
        return eps_pred, noise

    def shared_step(self, batch, split):
        z_seq = self.get_latents(batch)
        B = z_seq.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z_seq.device, dtype=torch.long)
        eps_pred, noise = self(z_seq, t)
        loss = F.mse_loss(eps_pred, noise)
        self.log(
            f"{split}/loss_simple",
            loss,
            prog_bar=True,
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            sync_dist=(split == "val"),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def extract_semantic(self, z_seq):
        return self.model.extract_semantic(z_seq)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        z_seq = self.get_latents(batch)
        B = z_seq.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z_seq.device, dtype=torch.long)
        eps_pred, noise = self(z_seq, t)
        z_noisy = self.schedule.q_sample(z_seq, t, noise)
        s1 = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        s2 = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        x0_hat = (z_noisy - s2 * eps_pred) / s1
        return {
            "context": self.decode_first_stage(z_seq[:, -2]),
            "target": self.decode_first_stage(z_seq[:, -1]),
            "denoise_x0": self.decode_first_stage(x0_hat[:, -1]),
        }
