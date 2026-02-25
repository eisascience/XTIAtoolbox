"""
tests/test_io_axes.py – unit tests for OME-TIFF detection and axes-to-RGB conversion.
"""
from __future__ import annotations

import numpy as np
import pytest

# Import helpers directly; skip if tifffile is absent
pytest.importorskip("tifffile")
import tifffile  # noqa: E402

from core.io import (
    _axes_to_rgb,
    _extract_channel_as_gray,
    _is_ome_tiff,
    _normalize_to_uint8,
    _tiff_to_rgb,
    get_channel_info,
)


# ---------------------------------------------------------------------------
# _axes_to_rgb tests
# ---------------------------------------------------------------------------

def test_yx_grayscale_to_rgb():
    data = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = _axes_to_rgb(data, "YX")
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8
    # All three channels should be identical (greyscale repeat)
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])


def test_yxc_3channel():
    data = np.zeros((16, 16, 3), dtype=np.uint8)
    data[..., 0] = 100
    data[..., 1] = 150
    data[..., 2] = 200
    rgb = _axes_to_rgb(data, "YXC")
    assert rgb.shape == (16, 16, 3)
    assert rgb[0, 0, 0] == 100
    assert rgb[0, 0, 1] == 150
    assert rgb[0, 0, 2] == 200


def test_yxc_1channel_to_rgb():
    data = np.full((8, 8, 1), 42, dtype=np.uint8)
    rgb = _axes_to_rgb(data, "YXC")
    assert rgb.shape == (8, 8, 3)
    assert (rgb == 42).all()


def test_cyx_3channel():
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    data[0] = 10
    data[1] = 20
    data[2] = 30
    rgb = _axes_to_rgb(data, "CYX")
    assert rgb.shape == (16, 16, 3)
    assert rgb[0, 0, 0] == 10
    assert rgb[0, 0, 1] == 20
    assert rgb[0, 0, 2] == 30


def test_cyx_1channel_to_rgb():
    data = np.full((1, 8, 8), 99, dtype=np.uint8)
    rgb = _axes_to_rgb(data, "CYX")
    assert rgb.shape == (8, 8, 3)
    assert (rgb == 99).all()


def test_zyx_takes_first_slice():
    data = np.zeros((4, 16, 16), dtype=np.uint8)
    data[0] = 55  # first slice
    data[1] = 77
    rgb = _axes_to_rgb(data, "ZYX")
    assert rgb.shape == (16, 16, 3)
    # First Z-slice is 55, converted to greyscale RGB
    assert (rgb == 55).all()


def test_zyxc_takes_first_slice():
    data = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    data[0, ..., 0] = 10
    data[0, ..., 1] = 20
    data[0, ..., 2] = 30
    rgb = _axes_to_rgb(data, "ZYXC")
    assert rgb.shape == (8, 8, 3)
    assert rgb[0, 0, 0] == 10


def test_uint16_normalized_to_uint8():
    data = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    rgb = _axes_to_rgb(data, "YX")
    assert rgb.dtype == np.uint8
    assert rgb.max() == 255
    assert rgb.min() == 0


def test_yxc_4channel_uses_first_3():
    data = np.zeros((8, 8, 4), dtype=np.uint8)
    data[..., :3] = np.array([5, 10, 15], dtype=np.uint8)
    data[..., 3] = 255  # alpha – should be discarded
    rgb = _axes_to_rgb(data, "YXC")
    assert rgb.shape == (8, 8, 3)
    assert rgb[0, 0, 0] == 5


# ---------------------------------------------------------------------------
# _is_ome_tiff tests
# ---------------------------------------------------------------------------

def test_ome_tiff_by_extension(tmp_path):
    p = tmp_path / "sample.ome.tif"
    p.write_bytes(b"\x00" * 16)
    assert _is_ome_tiff(p) is True

    p2 = tmp_path / "sample.ome.tiff"
    p2.write_bytes(b"\x00" * 16)
    assert _is_ome_tiff(p2) is True


def test_ome_tiff_by_magic_bytes(tmp_path):
    # Write a fake .tif with OME-XML magic in the header
    p = tmp_path / "sample.tif"
    p.write_bytes(b"II\x2a\x00" + b"\x00" * 100 + b"OME-XML" + b"\x00" * 100)
    assert _is_ome_tiff(p) is True


def test_plain_tiff_not_ome(tmp_path):
    p = tmp_path / "plain.tif"
    p.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    assert _is_ome_tiff(p) is False


def test_non_tiff_not_ome(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    assert _is_ome_tiff(p) is False


# ---------------------------------------------------------------------------
# _tiff_to_rgb round-trip via tifffile
# ---------------------------------------------------------------------------

def test_tiff_to_rgb_grayscale(tmp_path):
    path = tmp_path / "gray.tif"
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    tifffile.imwrite(str(path), arr)
    rgb = _tiff_to_rgb(path)
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8


def test_tiff_to_rgb_rgb_image(tmp_path):
    path = tmp_path / "rgb.tif"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 0] = 200
    tifffile.imwrite(str(path), arr)
    rgb = _tiff_to_rgb(path)
    assert rgb.shape == (16, 16, 3)
    assert rgb[0, 0, 0] == 200


# ---------------------------------------------------------------------------
# _extract_channel_as_gray
# ---------------------------------------------------------------------------

def test_extract_channel_cyx_ch0():
    data = np.zeros((3, 8, 8), dtype=np.uint8)
    data[0] = 10
    data[1] = 20
    data[2] = 30
    ch = _extract_channel_as_gray(data, "CYX", 0)
    assert ch.shape == (8, 8)
    assert (ch == 10).all()


def test_extract_channel_cyx_ch2():
    data = np.zeros((3, 8, 8), dtype=np.uint8)
    data[2] = 55
    ch = _extract_channel_as_gray(data, "CYX", 2)
    assert ch.shape == (8, 8)
    assert (ch == 55).all()


def test_extract_channel_cyx_clamps_out_of_range():
    """channel_idx beyond the last channel should clamp to last."""
    data = np.zeros((2, 8, 8), dtype=np.uint8)
    data[1] = 77
    ch = _extract_channel_as_gray(data, "CYX", 99)
    assert (ch == 77).all()


def test_extract_channel_yxc():
    data = np.zeros((8, 8, 4), dtype=np.uint8)
    data[..., 3] = 42
    ch = _extract_channel_as_gray(data, "YXC", 3)
    assert ch.shape == (8, 8)
    assert (ch == 42).all()


def test_extract_channel_strips_z_and_t():
    # Shape: T=2, Z=3, C=2, Y=8, X=8
    data = np.zeros((2, 3, 2, 8, 8), dtype=np.uint8)
    data[0, 0, 1] = 99   # T=0, Z=0, C=1
    ch = _extract_channel_as_gray(data, "TZCYX", 1)
    assert ch.shape == (8, 8)
    assert (ch == 99).all()


def test_extract_channel_yx_no_channel_dim():
    data = np.full((8, 8), 33, dtype=np.uint8)
    ch = _extract_channel_as_gray(data, "YX", 0)
    assert ch.shape == (8, 8)
    assert (ch == 33).all()


# ---------------------------------------------------------------------------
# _normalize_to_uint8
# ---------------------------------------------------------------------------

def test_normalize_uint8_full_range():
    data = np.arange(256, dtype=np.uint8)
    result = _normalize_to_uint8(data, 0, 100)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_normalize_flat_array_returns_zeros():
    data = np.full((4, 4), 128, dtype=np.uint16)
    result = _normalize_to_uint8(data)
    assert (result == 0).all()


def test_normalize_percentile_clipping():
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 100], dtype=np.float32)
    result = _normalize_to_uint8(data, low_pct=1, high_pct=90)
    assert result.dtype == np.uint8
    assert result[-1] == 255   # clipped to max


# ---------------------------------------------------------------------------
# get_channel_info
# ---------------------------------------------------------------------------

def test_get_channel_info_multichannel_tiff(tmp_path):
    path = tmp_path / "multichan.tif"
    arr = np.zeros((5, 16, 16), dtype=np.uint8)
    tifffile.imwrite(str(path), arr, imagej=True)
    n, names = get_channel_info(path)
    assert n == 5
    assert names == [f"Ch {i}" for i in range(5)]


def test_get_channel_info_grayscale_tiff(tmp_path):
    path = tmp_path / "gray.tif"
    arr = np.zeros((16, 16), dtype=np.uint8)
    tifffile.imwrite(str(path), arr)
    n, names = get_channel_info(path)
    assert n == 1
    assert names == ["Ch 0"]


def test_get_channel_info_non_tiff(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    n, names = get_channel_info(path)
    assert n == 1
    assert names == ["Ch 0"]
