"""
core/registry.py – persistent file registry stored as JSON on disk.
All mutations are written atomically (write-then-replace).
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from .config import REGISTRY_FILE
from .models import FileEntry

logger = logging.getLogger(__name__)


def _load_raw() -> dict[str, Any]:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with open(REGISTRY_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("registry load error: %s", exc)
        return {}


def _save_raw(data: dict[str, Any]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=REGISTRY_FILE.parent, prefix=".registry_", suffix=".tmp"
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        Path(tmp_path).replace(REGISTRY_FILE)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_files() -> list[FileEntry]:
    raw = _load_raw()
    entries = []
    for v in raw.values():
        try:
            entries.append(FileEntry.from_dict(v))
        except Exception as exc:
            logger.warning("Skipping corrupt registry entry: %s", exc)
    return entries


def get_file(file_id: str) -> FileEntry | None:
    raw = _load_raw()
    entry = raw.get(file_id)
    if entry is None:
        return None
    try:
        return FileEntry.from_dict(entry)
    except Exception as exc:
        logger.warning("Corrupt registry entry %s: %s", file_id, exc)
        return None


def add_file(entry: FileEntry) -> None:
    raw = _load_raw()
    raw[entry.file_id] = entry.as_dict()
    _save_raw(raw)


def remove_file(file_id: str) -> bool:
    raw = _load_raw()
    if file_id not in raw:
        return False
    del raw[file_id]
    _save_raw(raw)
    return True


def update_file(entry: FileEntry) -> None:
    raw = _load_raw()
    raw[entry.file_id] = entry.as_dict()
    _save_raw(raw)
