"""
S3 无监督序列数据集（filelist 驱动）：
从 dataset/filelist.txt 读取全部切片路径（xz+yz 双向窗口），按体名过滤、
按窗口文件夹分组，每个样本取文件夹内连续 context_window+1 张切片。
图像预处理与 AE 训练（taming ImagePaths）完全一致：
PIL → RGB → uint8 → SmallestMaxSize + CenterCrop → [-1,1] float32。
"""
import os
import re
from pathlib import Path

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 窗口文件夹名 = 体名 + 平面/窗口后缀：'fig3_denoise_xz00x00' → 'fig3_denoise'
_VOLUME_RE = re.compile(r"^(?P<vol>.+)_[xy]z\d+x\d+$")


def parse_volume_name(folder_name):
    """从窗口文件夹名解析体名；不匹配返回 None"""
    m = _VOLUME_RE.match(folder_name)
    return m.group("vol") if m is not None else None


class SeqSlicesBase(Dataset):
    def __init__(
        self,
        data_root,
        context_window=4,
        input_size=256,
        split_volumes=(),
        stride=1,
        max_windows_per_folder=None,
    ):
        self.data_root = data_root
        self.context_window = context_window
        self.input_size = input_size
        self.stride = stride
        self.seq_len = context_window + 1
        self.split_volumes = tuple(split_volumes)
        self.max_windows_per_folder = max_windows_per_folder

        # 与 taming ImagePaths 完全一致的预处理（AE 训练同款）
        self.preprocessor = albumentations.Compose(
            [
                albumentations.SmallestMaxSize(max_size=input_size),
                albumentations.CenterCrop(height=input_size, width=input_size),
            ]
        )
        self._prepare()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, start = self.samples[i]
        files = self.folder_files[name][start : start + self.seq_len]
        seq = np.stack([self._load_slice(p) for p in files])  # (L, H, W, 3)
        return {"seq": seq, "folder": name, "start": start}

    def _prepare(self):
        filelist_path = os.path.join(self.data_root, "dataset/filelist.txt")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")

        # filelist 每行是一条完整切片路径（repo 相对）。按窗口文件夹分组：
        # 切片的物理连续性只在单个窗口文件夹内成立，若按体名合并，同一体的
        # xz 序列末尾会与 yz 序列开头拼进同一个窗口。体名从文件夹名解析后过滤。
        self.folder_files = {}
        with open(filelist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                folder = p.parent.name
                if parse_volume_name(folder) not in self.split_volumes:
                    continue
                self.folder_files.setdefault(folder, []).append(p)

        # 文件夹内按文件名升序（零填充序号，等价数值序）
        for folder in self.folder_files:
            self.folder_files[folder].sort(key=lambda p: p.name)

        # 枚举滑窗起点；max_windows_per_folder 在起点范围内均匀抽稀
        # （linspace 恰好取 M 个且铺满整条序列，供 tiny 冒烟用"少而广"的子集）
        self.samples = []
        for name, files in self.folder_files.items():
            starts = list(range(0, len(files) - self.seq_len + 1, self.stride))
            if (
                self.max_windows_per_folder is not None
                and len(starts) > self.max_windows_per_folder
            ):
                sel = np.linspace(
                    0, len(starts) - 1, self.max_windows_per_folder
                ).round().astype(int)
                starts = [starts[i] for i in sel]
            self.samples.extend((name, s) for s in starts)

    def _load_slice(self, path):
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image


class SeqSlicesTrain(SeqSlicesBase):
    def __init__(self, data_root=None, train_volumes=(), **kwargs):
        super().__init__(data_root=data_root, split_volumes=train_volumes, **kwargs)


class SeqSlicesValidation(SeqSlicesBase):
    def __init__(self, data_root=None, val_volumes=(), **kwargs):
        super().__init__(data_root=data_root, split_volumes=val_volumes, **kwargs)
