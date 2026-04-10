import os

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from taming.data.base import ImagePaths


class FingerprintPaths(ImagePaths):
    """
    用于指纹B-scan图像的路径处理
    """

    def __init__(self, paths, labels=None):
        super().__init__(paths, size=0, random_crop=False, labels=labels)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # 归一化到[-1, 1]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image


class FingerprintBase(Dataset):
    """
    指纹B-scan体数据数据集基类

    沿x-z和y-z方向滑窗切分图像并保存
    """

    def __init__(
        self,
        config=None,
        size=256,
        data_root=None,
        split="train",
        train_ratio=0.8,
    ):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        self.size = size
        self.split = split
        self.train_ratio = train_ratio

        if data_root is not None:
            self.data_root = data_root
        else:
            self.data_root = "data/fingerprint-bscan"

        # 预处理（滑窗切分）
        self._prepare()
        # 加载图像路径
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        """
        滑窗切分体数据并保存为图像文件
        """
        raw_dir = os.path.join(self.data_root, "raw_data")
        self.cropped_dir = os.path.join(self.data_root, "cropped_images")
        prepare_flag = os.path.join(self.data_root, ".prepared")

        if not os.path.exists(prepare_flag):
            print(f"Preparing fingerprint bscan dataset from {raw_dir}")
            os.makedirs(self.cropped_dir, exist_ok=True)

            total_image_paths = []

            # 加载数据 (假设每个体数据放在独立的一个文件夹下)
            with os.scandir(raw_dir) as entries:
                volume_dirs = [f.name for f in entries if f.is_dir()]

            for volume_dir in volume_dirs:
                volume_path = os.path.join(raw_dir, volume_dir)
                npy_files = [f for f in os.listdir(volume_path) if f.endswith(".npy")]
                if len(npy_files) == 0:
                    print(f"No .npy files found in {volume_path}")
                    continue

                depth, width = np.load(os.path.join(volume_path, npy_files[0])).shape  # 假设bscan的尺寸都相同
                height = len(npy_files)
                volume = np.zeros((height, depth, width))
                print(f"Building volume: {volume_dir}")
                for i, npy_file in enumerate(npy_files):
                    npy_path = os.path.join(volume_path, npy_file)
                    bscan = np.load(npy_path)
                    d, w = bscan.shape
                    d_min, w_min = min(depth, d), min(width, w)
                    volume[i, :d_min, :w_min] = bscan[:d_min, :w_min]

                # 归一化到[0, 255]
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
                volume = (volume * 255).astype(np.uint8)

                crop_size = self.size
                stride = crop_size // 2

                print(f"Processing...")
                x_positions = self._get_window_positions(width, crop_size, stride)  # x方向窗口位置
                y_positions = self._get_window_positions(height, crop_size, stride)  # y方向窗口位置
                z_positions = self._get_window_positions(depth, crop_size, stride)  # z方向窗口位置

                image_paths = []

                # 1. x-z面切分
                for xi, x_start in enumerate(x_positions):
                    for zi, z_start in enumerate(z_positions):
                        for y in range(height):
                            patch = volume[y, z_start : z_start + crop_size, x_start : x_start + crop_size]

                            # 保存为图像
                            sub_save_path = os.path.join(self.cropped_dir, f"{volume_dir}_x{xi:02d}_z{zi:02d}")
                            os.makedirs(sub_save_path, exist_ok=True)
                            save_path = os.path.join(sub_save_path, f"{y:04d}.jpg")

                            if not os.path.exists(save_path):
                                img = Image.fromarray(patch, mode="L").convert("RGB")
                                img.save(save_path)

                            image_paths.append(save_path)

                # 2. y-z面切分
                for yi, y_start in enumerate(y_positions):
                    for zi, z_start in enumerate(z_positions):
                        for x in range(width):
                            patch = volume[y_start : y_start + crop_size, z_start : z_start + crop_size, x]

                            # 保存为图像
                            sub_save_path = os.path.join(self.cropped_dir, f"{volume_dir}_y{yi:02d}_z{zi:02d}")
                            os.makedirs(sub_save_path, exist_ok=True)
                            save_path = os.path.join(sub_save_path, f"{x:04d}.jpg")

                            if not os.path.exists(save_path):
                                img = Image.fromarray(patch, mode="L").convert("RGB")
                                img.save(save_path)

                            image_paths.append(save_path)

                print(f"  Generated {len(image_paths)} patches from {volume_dir}")
                total_image_paths.extend(image_paths)

            # 保存文件列表
            filelist_path = os.path.join(self.data_root, "filelist.txt")
            with open(filelist_path, "w") as f:
                for path in total_image_paths:
                    # 保存相对路径
                    rel_path = os.path.relpath(path, self.cropped_dir)
                    f.write(rel_path + "\n")

            # 标记预处理完成
            with open(prepare_flag, "w") as f:
                f.write("prepared")

            print(f"Preparation complete. Total patches: {len(total_image_paths)}")

    def _get_window_positions(self, length, crop_size, stride):
        """
        计算滑窗位置
        """
        positions = []
        pos = 0
        while pos + crop_size <= length:
            positions.append(pos)
            pos += stride

        # 添加覆盖末尾的最后一个窗口
        # if positions[-1] + crop_size < length:
        #     positions.append(length - crop_size)

        return positions

    def _load(self):
        """
        加载文件列表并创建数据集
        """
        filelist_path = os.path.join(self.data_root, "filelist.txt")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")

        with open(filelist_path, "r") as f:
            relpaths = f.read().splitlines()

        # 划分训练集/验证集
        all_paths = [os.path.join(self.cropped_dir, p) for p in relpaths]
        n_total = len(all_paths)
        n_train = int(n_total * self.train_ratio)

        if self.split == "train":
            split_paths = all_paths[:n_train]
        else:
            split_paths = all_paths[n_train:]

        print(f"Split: {self.split}, Images: {len(split_paths)}/{n_total}")

        # 创建labels
        labels = {
            "relpath": np.array([os.path.relpath(p, self.cropped_dir) for p in split_paths]),
            "file_path_": np.array(split_paths),
        }

        self.data = FingerprintPaths(
            split_paths,
            labels=labels,
        )


class FingerprintTrain(FingerprintBase):
    """
    训练集
    """

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        super().__init__(data_root=data_root, split="train", **kwargs)


class FingerprintValidation(FingerprintBase):
    """
    验证集
    """

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        super().__init__(data_root=data_root, split="validation", **kwargs)
