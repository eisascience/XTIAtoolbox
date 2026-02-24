"""
core/models.py – Pydantic-free data models (plain dataclasses) for the orchestrator.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# File entry (uploaded slide or image)
# ---------------------------------------------------------------------------
@dataclass
class FileEntry:
    file_id: str
    original_name: str
    stored_path: Path
    sha256: str
    size_bytes: int
    uploaded_at: str  # ISO-8601
    is_wsi: bool = False
    width: int = 0
    height: int = 0
    mpp: float = 0.0   # microns per pixel (0 if unknown)
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["stored_path"] = str(self.stored_path)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileEntry":
        d = d.copy()
        d["stored_path"] = Path(d["stored_path"])
        if "tags" not in d:
            d["tags"] = []
        return cls(**d)


# ---------------------------------------------------------------------------
# ROI definition
# ---------------------------------------------------------------------------
@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int
    level: int = 0

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return (x, y, x+w, y+h)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------
@dataclass
class RunManifest:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task_name: str = ""
    task_params: dict[str, Any] = field(default_factory=dict)
    input_files: list[dict[str, Any]] = field(default_factory=list)  # [{name, sha256, size}]
    roi: dict[str, Any] | None = None
    device: str = "cpu"
    tiatoolbox_version: str = ""
    python_version: str = ""
    outputs: list[str] = field(default_factory=list)   # relative paths under run dir
    status: str = "pending"   # pending | running | completed | failed
    error: str = ""
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        return cls(**d)


# ---------------------------------------------------------------------------
# Batch job entry
# ---------------------------------------------------------------------------
@dataclass
class BatchJob:
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    file_id: str = ""
    task_name: str = ""
    task_params: dict[str, Any] = field(default_factory=dict)
    roi: dict[str, Any] | None = None
    status: str = "queued"   # queued | running | completed | failed
    run_id: str = ""
    error: str = ""
    added_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatchJob":
        return cls(**d)
