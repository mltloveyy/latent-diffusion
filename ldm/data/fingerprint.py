import os
import random

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from taming.data.base import ImagePaths


class FingerprintPaths(ImagePaths):
    """
    用于切片图像预处理
    """

    def __init__(self, paths, size, labels=None):
        super().__init__(paths, size, False, labels)


class FingerprintBase(Dataset):
    """
    将指纹B-scan数据合成体数据，并分别基于x-z和y-z面切片后保存为图像
    """

    def __init__(
        self,
        config=None,
        window_size=256,
        input_size=256,
        data_root=None,
        train_ratio=0.8,
        split="train",
    ):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        self.window_size = window_size
        self.input_size = input_size
        self.train_ratio = train_ratio
        self.data_root = data_root
        self.split = split

        self.raw_path = os.path.join(self.data_root, "raw")
        self.dataset_path = os.path.join(self.data_root, "dataset")
        os.makedirs(self.dataset_path, exist_ok=True)
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        """
        图像堆叠成体数据，再沿着x和y方向滑窗切片
        """
        filelist_path = os.path.join(self.dataset_path, "filelist.txt")
        if os.path.exists(filelist_path):
            print(f"Preprocessed data found at {filelist_path}, skipping preparation.")
            return

        def _load_bscan(path: str):
            if path.endswith(".npy"):
                img = np.load(path)
            elif path.endswith(".png") | path.endswith(".jpg"):
                img = np.array(Image.open(path))
            else:
                raise ValueError(f"{path[-4:]} format is not support")
            height, width = img.shape
            return img, height, width

        def _get_window_positions(length, window_size, stride):
            positions = []
            pos = 0
            while pos + window_size <= length:
                positions.append(pos)
                pos += stride
            # if positions[-1] + window_size < length:
            #     positions.append(length - window_size)  # 添加覆盖末尾的最后一个窗口
            return positions

        def _slice(volume, plane, positions1, positions2, length, window_size, prefix):
            """
            体数据切片成图像
            """
            img_paths = []
            for i, pos1 in enumerate(positions1):
                for j, pos2 in enumerate(positions2):
                    for k in range(length):
                        if plane == "xz":
                            slice_patch = volume[pos2 : pos2 + window_size, pos1 : pos1 + window_size, k]
                        else:
                            slice_patch = volume[pos2 : pos2 + window_size, k, pos1 : pos1 + window_size]

                        save_dir = prefix + f"_{plane}{i:02d}x{j:02d}"
                        os.makedirs(save_dir, exist_ok=True)
                        img_path = os.path.join(save_dir, f"{k:04d}.png")
                        img = Image.fromarray(slice_patch, mode="L")
                        img.save(img_path)
                        img_paths.append(img_path)
            return img_paths

        with os.scandir(self.raw_path) as entries:
            sub_dirs = [f.name for f in entries if f.is_dir()]

        total_image_paths = []
        for sub_dir in sub_dirs:
            sub_path = os.path.join(self.raw_path, sub_dir)

            # 图像堆叠成体数据
            name_list = os.listdir(sub_path)
            _, depth, width = _load_bscan(os.path.join(sub_path, name_list[0]))
            length = len(name_list)
            volume = np.zeros((depth, width, length))
            print(f"Stacking volume: {sub_path}")
            for i, name in enumerate(name_list):
                bscan, _, _ = _load_bscan(os.path.join(sub_path, name))
                volume[:, :, i] = bscan  # 假设所有bscan的尺寸都相同
            volume = volume.astype(np.uint8)

            x_positions = _get_window_positions(width, self.window_size, self.window_size // 2)  # x方向窗口位置
            y_positions = _get_window_positions(length, self.window_size, self.window_size // 2)  # y方向窗口位置
            z_positions = _get_window_positions(depth, self.window_size, self.window_size // 2)  # z方向窗口位置

            print(f"Volume shape: ({depth}, {length}, {width}). Slicing...")
            prefix = os.path.join(self.dataset_path, sub_dir)
            img_paths_xz = _slice(volume, "xz", x_positions, z_positions, length, self.window_size, prefix)
            img_paths_yz = _slice(volume, "yz", y_positions, z_positions, width, self.window_size, prefix)
            image_paths = img_paths_xz + img_paths_yz
            print(f"Generated {len(image_paths)} patches from {sub_dir}")
            total_image_paths.extend(image_paths)

        # 保存文件列表
        with open(filelist_path, "w", encoding="utf-8") as f:
            f.write("\n".join(total_image_paths))
        print(f"Preparation complete. Total patches: {len(total_image_paths)}")

    def _load(self):
        filelist_path = os.path.join(self.dataset_path, "filelist.txt")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")

        with open(filelist_path, "r") as f:
            all_paths = f.read().splitlines()

        # 划分训练集/验证集
        random.shuffle(all_paths)
        n_total = len(all_paths)
        n_train = int(n_total * self.train_ratio)
        if self.split == "train":
            split_paths = all_paths[:n_train]
        else:
            split_paths = all_paths[n_train:]
        print(f"Split: {self.split}, Images: {len(split_paths)}/{n_total}")

        # 创建labels
        labels = {
            "relpath": np.array([os.path.relpath(p, self.dataset_path) for p in split_paths]),
            "file_path_": np.array(split_paths),
        }
        self.data = FingerprintPaths(split_paths, size=self.input_size, labels=labels)


class FingerprintTrain(FingerprintBase):
    def __init__(self, data_root=None, **kwargs):
        super().__init__(data_root=data_root, split="train", **kwargs)


class FingerprintValidation(FingerprintBase):
    def __init__(self, data_root=None, **kwargs):
        super().__init__(data_root=data_root, split="validation", **kwargs)
