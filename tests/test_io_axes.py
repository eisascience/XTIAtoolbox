"""
tests/test_io_axes.py – unit tests for OME-TIFF detection and axes-to-RGB conversion.
"""
from __future__ import annotations

import numpy as np
import pytest

# Import helpers directly; skip if tifffile is absent
pytest.importorskip("tifffile")
import tifffile  # noqa: E402

from core.io import _axes_to_rgb, _is_ome_tiff, _tiff_to_rgb


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
