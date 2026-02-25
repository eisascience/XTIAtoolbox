"""
tests/test_channel_config.py – unit tests for core/channel_config.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.channel_config import (
    APP_VERSION,
    DEFAULT_CHANNEL_CONFIG,
    build_config_json,
    config_filename,
    get_channel_percentiles,
    load_config_from_disk,
    load_config_json,
    save_config_to_disk,
)


# ---------------------------------------------------------------------------
# config_filename
# ---------------------------------------------------------------------------

def test_config_filename_plain():
    assert config_filename("image.tiff") == "image.tiff.xhisto_channel_config.json"


def test_config_filename_ome():
    fn = "020526_RNo18.ome.tiff"
    assert config_filename(fn) == fn + ".xhisto_channel_config.json"


# ---------------------------------------------------------------------------
# build_config_json
# ---------------------------------------------------------------------------

def test_build_config_json_basic():
    doc = build_config_json("img.tiff", 3, ["DAPI", "CD5", "vRNA"])
    assert doc["original_filename"] == "img.tiff"
    assert doc["channel_count"] == 3
    assert doc["channel_names"] == ["DAPI", "CD5", "vRNA"]
    assert doc["app_version"] == APP_VERSION
    assert "timestamp" in doc
    assert doc["channel_configs_by_index"] == {}


def test_build_config_json_with_configs():
    cfgs = {"0": {"percentile_low": 2.0, "percentile_high": 98.0}}
    doc = build_config_json("img.tiff", 2, ["A", "B"], channel_configs_by_index=cfgs)
    assert doc["channel_configs_by_index"] == cfgs


def test_build_config_json_passthrough_unknown_keys():
    existing = {
        "original_filename": "old.tiff",
        "channel_count": 1,
        "channel_names": ["Ch 0"],
        "channel_configs_by_index": {},
        "timestamp": "2024-01-01T00:00:00",
        "app_version": "1.0",
        "my_custom_key": "preserve_me",
        "another_custom": 42,
    }
    doc = build_config_json(
        "new.tiff", 2, ["X", "Y"], existing_data=existing
    )
    # Known keys are updated
    assert doc["original_filename"] == "new.tiff"
    assert doc["channel_count"] == 2
    assert doc["channel_names"] == ["X", "Y"]
    # Unknown keys are preserved
    assert doc["my_custom_key"] == "preserve_me"
    assert doc["another_custom"] == 42


def test_build_config_json_keeps_existing_channel_configs_when_none_supplied():
    existing = {
        "original_filename": "img.tiff",
        "channel_count": 2,
        "channel_names": ["A", "B"],
        "channel_configs_by_index": {"0": {"percentile_low": 5.0}},
        "timestamp": "2024-01-01T00:00:00",
        "app_version": "1.0",
    }
    doc = build_config_json("img.tiff", 2, ["A", "B"], existing_data=existing)
    assert doc["channel_configs_by_index"] == {"0": {"percentile_low": 5.0}}


# ---------------------------------------------------------------------------
# load_config_json
# ---------------------------------------------------------------------------

def test_load_config_json_full_schema():
    """Round-trip the exact reference schema from the problem statement."""
    raw = {
        "original_filename": "020526_RNo18.ome.tiff",
        "channel_count": 5,
        "channel_names": ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4"],
        "channel_configs_by_index": {
            "0": {
                "use_normalization": True,
                "percentile_low": 1.0,
                "percentile_high": 99.0,
                "threshold_mode": "off",
                "threshold_value": 128,
                "threshold_type": "binary",
                "smoothing_sigma": 0.0,
            }
        },
        "timestamp": "2026-02-20T23:10:22.754191",
        "app_version": "1.0",
    }
    orig_fn, ch_count, ch_names, ch_cfgs, passthrough = load_config_json(raw)
    assert orig_fn == "020526_RNo18.ome.tiff"
    assert ch_count == 5
    assert ch_names == ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4"]
    assert ch_cfgs["0"]["percentile_low"] == 1.0
    assert ch_cfgs["0"]["threshold_mode"] == "off"
    assert passthrough == {}


def test_load_config_json_passthrough():
    raw = {
        "original_filename": "img.tiff",
        "channel_count": 1,
        "channel_names": ["DAPI"],
        "channel_configs_by_index": {},
        "timestamp": "2024-01-01",
        "app_version": "1.0",
        "extra_tool_key": "foo",
        "another": [1, 2, 3],
    }
    _, _, _, _, passthrough = load_config_json(raw)
    assert passthrough == {"extra_tool_key": "foo", "another": [1, 2, 3]}


def test_load_config_json_missing_keys_use_defaults():
    orig_fn, ch_count, ch_names, ch_cfgs, passthrough = load_config_json({})
    assert orig_fn == ""
    assert ch_count == 0
    assert ch_names == []
    assert ch_cfgs == {}
    assert passthrough == {}


# ---------------------------------------------------------------------------
# Round-trip: build → JSON string → load
# ---------------------------------------------------------------------------

def test_round_trip_preserves_all_fields():
    original = {
        "original_filename": "slide.ome.tiff",
        "channel_count": 3,
        "channel_names": ["DAPI", "FITC", "TRITC"],
        "channel_configs_by_index": {
            "0": dict(DEFAULT_CHANNEL_CONFIG),
            "1": dict(DEFAULT_CHANNEL_CONFIG),
        },
        "timestamp": "2025-01-01T00:00:00",
        "app_version": "1.0",
        "lab_id": "LAB42",
    }
    # Build a new doc updating names but preserving passthrough
    doc = build_config_json(
        "slide.ome.tiff", 3, ["Nucleus", "GFP", "RFP"], existing_data=original
    )
    # Serialize and deserialize
    reloaded = json.loads(json.dumps(doc))
    orig_fn, ch_count, ch_names, ch_cfgs, passthrough = load_config_json(reloaded)
    assert orig_fn == "slide.ome.tiff"
    assert ch_count == 3
    assert ch_names == ["Nucleus", "GFP", "RFP"]
    assert "0" in ch_cfgs
    assert passthrough == {"lab_id": "LAB42"}


# ---------------------------------------------------------------------------
# get_channel_percentiles
# ---------------------------------------------------------------------------

def test_get_channel_percentiles_present():
    cfgs = {"2": {"percentile_low": 5.0, "percentile_high": 95.0}}
    lo, hi = get_channel_percentiles(cfgs, 2)
    assert lo == 5.0
    assert hi == 95.0


def test_get_channel_percentiles_missing_uses_default():
    lo, hi = get_channel_percentiles({}, 0)
    assert lo == DEFAULT_CHANNEL_CONFIG["percentile_low"]
    assert hi == DEFAULT_CHANNEL_CONFIG["percentile_high"]


# ---------------------------------------------------------------------------
# save_config_to_disk / load_config_from_disk
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    orig = "my_image.ome.tiff"
    names = ["DAPI", "CD5", "vRNA"]
    cfgs = {"0": dict(DEFAULT_CHANNEL_CONFIG)}

    out_path = save_config_to_disk(tmp_path, orig, 3, names, cfgs)
    assert out_path.exists()
    assert out_path.name == config_filename(orig)

    loaded = load_config_from_disk(tmp_path, orig)
    assert loaded is not None
    assert loaded["original_filename"] == orig
    assert loaded["channel_count"] == 3
    assert loaded["channel_names"] == names
    assert loaded["channel_configs_by_index"] == cfgs


def test_load_config_from_disk_missing_returns_none(tmp_path):
    result = load_config_from_disk(tmp_path, "nonexistent.tiff")
    assert result is None


def test_save_preserves_passthrough_keys(tmp_path):
    existing = {
        "original_filename": "img.tiff",
        "channel_count": 1,
        "channel_names": ["Ch 0"],
        "channel_configs_by_index": {},
        "timestamp": "2024-01-01",
        "app_version": "1.0",
        "custom_field": "should_survive",
    }
    save_config_to_disk(
        tmp_path, "img.tiff", 1, ["DAPI"],
        existing_data=existing,
    )
    loaded = load_config_from_disk(tmp_path, "img.tiff")
    assert loaded["custom_field"] == "should_survive"
    assert loaded["channel_names"] == ["DAPI"]
