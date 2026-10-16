import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())

from ldm.data.seq_slices import SeqSlicesBase, SeqSlicesTrain, SeqSlicesValidation, parse_volume_name

DATA_ROOT = "data/fingerprint/UESTC"
requires_data = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA_ROOT, "dataset")), reason="UESTC 数据不存在")


def test_parse_volume_name():
    assert parse_volume_name("fig3_denoise_xz00x00") == "fig3_denoise"
    assert parse_volume_name("fig4_5_p0004_yz00x00") == "fig4_5_p0004"
    assert parse_volume_name("fig8_original_oct_image_xz00x00") == "fig8_original_oct_image"
    assert parse_volume_name("filelist.txt") is None


@requires_data
def test_seq_dataset_length_formula():
    # fig3_denoise 有 xz/yz 两个窗口文件夹，每个 800 张，L=5 → 每文件夹 796 个起点
    ds = SeqSlicesBase(DATA_ROOT, context_window=4, split_volumes=("fig3_denoise",))
    assert len(ds) == 796 * 2


@requires_data
def test_seq_dataset_shapes_and_range():
    ds = SeqSlicesBase(DATA_ROOT, context_window=4, split_volumes=("fig3_denoise",),
                       max_windows_per_folder=2)
    assert len(ds) == 4
    item = ds[0]
    assert item["seq"].shape == (5, 256, 256, 3)
    assert item["seq"].dtype == np.float32
    assert item["seq"].min() >= -1.0 and item["seq"].max() <= 1.0
    assert isinstance(item["folder"], str) and isinstance(item["start"], int)


@requires_data
def test_seq_dataset_includes_both_planes():
    ds = SeqSlicesBase(DATA_ROOT, context_window=4, split_volumes=("fig3_denoise",))
    planes = {f.rsplit("_", 1)[-1][:2] for f, _ in ds.samples}   # 'xz' / 'yz'（取末段前2字符，体名可含下划线）
    assert planes == {"xz", "yz"}


@requires_data
def test_train_val_disjoint():
    tr = SeqSlicesTrain(DATA_ROOT, context_window=4, max_windows_per_folder=2,
                        train_volumes=["fig3_denoise", "fig8_original_oct_image", "fig4_5_p0004"])
    va = SeqSlicesValidation(DATA_ROOT, context_window=4, max_windows_per_folder=2,
                             val_volumes=["fig4_5_p0013"])
    tr_vols = {parse_volume_name(f) for f, _ in tr.samples}
    va_vols = {parse_volume_name(f) for f, _ in va.samples}
    assert tr_vols == {"fig3_denoise", "fig8_original_oct_image", "fig4_5_p0004"}
    assert va_vols == {"fig4_5_p0013"}
    assert tr_vols.isdisjoint(va_vols)


@requires_data
def test_stride_limits_windows():
    ds = SeqSlicesBase(DATA_ROOT, context_window=4, split_volumes=("fig3_denoise",), stride=100)
    assert len(ds) == 8 * 2     # 起点 0,100,...,700 → 8 个/文件夹
