
## 环境配置

- torch
- torchvision
- lightning
- einops
- omegaconf
- requests
- albumentations

## 数据集准备

### 数据集结构

```
data_root
├── raw/
│   ├── volume001/
│   │   ├── volume001_0000.png
│   │   ├── volume001_0001.png
│   │   └── ...
│   ├── volume002/
│   └── ...
├── dataset/
│   ├── volume001_xz00x00/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── volume001_xz00x01/
│   ├── volume001_xz01x00/
│   ├── volume001_xz01x01/
│   ├── volume001_yz00x00/
│   ├── volume002_xz00x00/
|   └── ...
└── filelist.txt
```

### 滑窗策略

假设每个体数据由1600张1800×500 的B-scan图像组成，即`x=1800, y=1600, z=500, win_size=256, stride=128`

- 窗口数 `win_num = round((length - win_size) / stride) + 1`

#### x-z面切片

- 沿x和z方向滑窗
- x方向: 13个窗口
- z方向: 2个窗口
- 生成 26 个子体数据，每个子体数据包含 1600 张 256×256 图像，总计 41600 张

#### y-z面切分

- 沿y和z方向滑窗
- y方向: 11个窗口
- z方向: 2个窗口
- 生成 22 个子体数据，每个子体数据包含 1800 张 256×256 图像，总计 39600 张

**单个体数据总计: 81200张 256×256 图像**

## 开发日志

- `2026.3.12`: 添加taming-transformers库，适配lightning==2.x
- `2026.4.10`: fingerprint-scan数据类实现
- `2026.8.11`: 优化fingerprint数据类实现(适配UESTC数据集)，跑通AutoencoderKL训练