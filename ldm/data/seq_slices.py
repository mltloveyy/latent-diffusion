import os
import re
from pathlib import Path

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_VOLUME_RE = re.compile(r"^(?P<vol>.+)_[xy]z\d+x\d+$")


def parse_volume_name(folder_name):
    """
    从窗口文件夹名解析体名；不匹配返回 None
    """
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

        for folder in self.folder_files:
            self.folder_files[folder].sort(key=lambda p: p.name)

        self.samples = []
        for name, files in self.folder_files.items():
            starts = list(range(0, len(files) - self.seq_len + 1, self.stride))
            if self.max_windows_per_folder is not None and len(starts) > self.max_windows_per_folder:
                sel = np.linspace(0, len(starts) - 1, self.max_windows_per_folder).round().astype(int)
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
