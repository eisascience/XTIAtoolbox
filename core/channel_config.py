"""
core/channel_config.py – helpers for the shared xhisto channel-config JSON format.

Schema (reference: *.xhisto_channel_config.json):
{
  "original_filename": "<image filename>",
  "channel_count": <int>,
  "channel_names": ["Ch 0", "Ch 1", ...],
  "channel_configs_by_index": {
    "0": {
      "use_normalization": true,
      "percentile_low": 1.0,
      "percentile_high": 99.0,
      "threshold_mode": "off",
      "threshold_value": 128,
      "threshold_type": "binary",
      "smoothing_sigma": 0.0
    }
  },
  "timestamp": "<ISO-8601>",
  "app_version": "1.0"
}

Unknown top-level keys are preserved on load and re-emitted on save (passthrough).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

APP_VERSION = "1.0"
_CONFIG_SUFFIX = ".xhisto_channel_config.json"

# Keys we know about – everything else is passthrough
_KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "original_filename",
        "channel_count",
        "channel_names",
        "channel_configs_by_index",
        "timestamp",
        "app_version",
    }
)

# Default per-channel config matching the reference schema
DEFAULT_CHANNEL_CONFIG: dict[str, Any] = {
    "use_normalization": True,
    "percentile_low": 1.0,
    "percentile_high": 99.0,
    "threshold_mode": "off",
    "threshold_value": 128,
    "threshold_type": "binary",
    "smoothing_sigma": 0.0,
}


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------

def config_filename(original_name: str) -> str:
    """Return the JSON config filename for a given image filename.

    Convention: ``<original_filename>.xhisto_channel_config.json``
    """
    return original_name + _CONFIG_SUFFIX


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_config_json(
    original_filename: str,
    channel_count: int,
    channel_names: list[str],
    channel_configs_by_index: dict[str, Any] | None = None,
    existing_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a channel-config dict ready for JSON serialisation.

    If *existing_data* is provided, unknown keys from it are preserved
    (passthrough) while known keys are updated from the supplied arguments.
    If *channel_configs_by_index* is None and *existing_data* contains one,
    the existing per-channel configs are kept.
    """
    doc: dict[str, Any] = {}

    # Passthrough: copy unknown keys from existing data first
    if existing_data:
        for k, v in existing_data.items():
            if k not in _KNOWN_KEYS:
                doc[k] = v

    resolved_configs: dict[str, Any]
    if channel_configs_by_index is not None:
        resolved_configs = channel_configs_by_index
    elif existing_data and "channel_configs_by_index" in existing_data:
        resolved_configs = existing_data["channel_configs_by_index"]
    else:
        resolved_configs = {}

    doc.update(
        {
            "original_filename": original_filename,
            "channel_count": channel_count,
            "channel_names": list(channel_names),
            "channel_configs_by_index": resolved_configs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_version": APP_VERSION,
        }
    )
    return doc


def load_config_json(
    data: dict[str, Any],
) -> tuple[str, int, list[str], dict[str, Any], dict[str, Any]]:
    """Parse a channel-config dict.

    Returns:
        ``(original_filename, channel_count, channel_names,
           channel_configs_by_index, passthrough)``

    *passthrough* contains all top-level keys that are not part of the
    known schema so they can be round-tripped when saving back.
    """
    passthrough = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    return (
        str(data.get("original_filename", "")),
        int(data.get("channel_count", 0)),
        list(data.get("channel_names", [])),
        dict(data.get("channel_configs_by_index", {})),
        passthrough,
    )


def get_channel_percentiles(
    channel_configs_by_index: dict[str, Any], channel_idx: int
) -> tuple[float, float]:
    """Return ``(percentile_low, percentile_high)`` for *channel_idx*.

    Falls back to ``DEFAULT_CHANNEL_CONFIG`` values if the index is absent.
    """
    cfg = channel_configs_by_index.get(str(channel_idx), DEFAULT_CHANNEL_CONFIG)
    return (
        float(cfg.get("percentile_low", DEFAULT_CHANNEL_CONFIG["percentile_low"])),
        float(cfg.get("percentile_high", DEFAULT_CHANNEL_CONFIG["percentile_high"])),
    )


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------

def save_config_to_disk(
    dest_dir: Path,
    original_filename: str,
    channel_count: int,
    channel_names: list[str],
    channel_configs_by_index: dict[str, Any] | None = None,
    existing_data: dict[str, Any] | None = None,
) -> Path:
    """Write (or overwrite) the channel config JSON in *dest_dir*.

    Returns the path that was written.
    """
    doc = build_config_json(
        original_filename,
        channel_count,
        channel_names,
        channel_configs_by_index,
        existing_data,
    )
    out_path = dest_dir / config_filename(original_filename)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    logger.debug("Saved channel config to %s", out_path)
    return out_path


def load_config_from_disk(
    upload_dir: Path, original_filename: str
) -> dict[str, Any] | None:
    """Load an existing channel config JSON from *upload_dir*, or ``None``."""
    config_path = upload_dir / config_filename(original_filename)
    if not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load channel config from %s: %s", config_path, exc)
        return None
