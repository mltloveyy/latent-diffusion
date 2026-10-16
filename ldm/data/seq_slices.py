"""
S3 无监督序列数据集：从 dataset/<体>_<平面>窗口/ 文件夹取连续 context_window+1 张切片。
图像预处理与 AE 训练（taming ImagePaths）完全一致：
PIL → RGB → uint8 → SmallestMaxSize + CenterCrop → [-1,1] float32。
xz 与 yz 两个方向的窗口文件夹均参与（专利 S1：双向拆解互为补充）。
"""
import os
import re

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_VOLUME_RE = re.compile(r"^(?P<vol>.+)_(?P<plane>[xy]z)(?P<w1>\d+)x(?P<w2>\d+)$")


def parse_volume_name(folder_name):
    """'fig3_denoise_xz00x00' → 'fig3_denoise'；不匹配返回 None"""
    m = _VOLUME_RE.match(folder_name)
    return m.group("vol") if m is not None else None


class SeqSlicesBase(Dataset):
    def __init__(self, data_root, context_window=4, input_size=256, split_volumes=(),
                 stride=1, max_windows_per_folder=None):
        self.data_root = data_root
        self.context_window = context_window
        self.input_size = input_size
        self.seq_len = context_window + 1
        self.split_volumes = tuple(split_volumes)

        self.preprocessor = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=input_size),
            albumentations.CenterCrop(height=input_size, width=input_size),
        ])

        dataset_path = os.path.join(data_root, "dataset")
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"dataset 目录不存在: {dataset_path}")

        self.folder_files = {}
        for name in sorted(os.listdir(dataset_path)):
            folder = os.path.join(dataset_path, name)
            if not os.path.isdir(folder):
                continue
            vol = parse_volume_name(name)
            if vol is None or vol not in self.split_volumes:
                continue
            files = sorted(os.listdir(folder))          # 0000.png...0799.png，零填充可直接排序
            if len(files) < self.seq_len:
                continue
            self.folder_files[name] = [os.path.join(folder, f) for f in files]

        # 枚举滑窗起点
        self.samples = []
        for name, files in self.folder_files.items():
            starts = list(range(0, len(files) - self.seq_len + 1, stride))
            if max_windows_per_folder is not None and len(starts) > max_windows_per_folder:
                sel = np.linspace(0, len(starts) - 1, max_windows_per_folder).round().astype(int)
                starts = [starts[i] for i in sel]
            self.samples.extend((name, s) for s in starts)

    def __len__(self):
        return len(self.samples)

    def _load_slice(self, path):
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        return (image / 127.5 - 1.0).astype(np.float32)

    def __getitem__(self, i):
        name, start = self.samples[i]
        files = self.folder_files[name][start:start + self.seq_len]
        seq = np.stack([self._load_slice(p) for p in files])    # (L, H, W, 3)
        return {"seq": seq, "folder": name, "start": start}


class SeqSlicesTrain(SeqSlicesBase):
    def __init__(self, data_root=None, train_volumes=(), **kwargs):
        super().__init__(data_root=data_root, split_volumes=train_volumes, **kwargs)


class SeqSlicesValidation(SeqSlicesBase):
    def __init__(self, data_root=None, val_volumes=(), **kwargs):
        super().__init__(data_root=data_root, split_volumes=val_volumes, **kwargs)
