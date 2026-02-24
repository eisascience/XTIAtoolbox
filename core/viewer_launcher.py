"""
core/viewer_launcher.py – launch the TIAToolbox Bokeh visualisation server.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_viewer_command(run_dir: Path) -> list[str]:
    """Return the command-line list that starts the TIAToolbox viewer."""
    out_dir = run_dir / "outputs"
    source_dir = str(out_dir) if out_dir.exists() else str(run_dir)

    # Try to locate tiatoolbox executable
    tb_exe = shutil.which("tiatoolbox") or sys.executable + " -m tiatoolbox"
    if shutil.which("tiatoolbox"):
        cmd = ["tiatoolbox", "visualize", "--img-input", source_dir]
    else:
        cmd = [sys.executable, "-m", "tiatoolbox", "visualize", "--img-input", source_dir]
    return cmd


def launch_viewer(run_dir: Path) -> tuple[bool, str]:
    """
    Attempt to launch the TIAToolbox viewer as a background subprocess.

    Returns (success, message).
    """
    cmd = build_viewer_command(run_dir)
    cmd_str = " ".join(cmd)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        return True, (
            f"Viewer launched (PID {proc.pid}).\n"
            f"Command: {cmd_str}\n"
            f"Open http://localhost:5006 in your browser."
        )
    except FileNotFoundError:
        return False, (
            f"Could not find tiatoolbox executable.\n"
            f"Run manually:\n    {cmd_str}"
        )
    except Exception as exc:
        logger.exception("launch_viewer failed")
        return False, f"Launch failed: {exc}\n\nRun manually:\n    {cmd_str}"


def viewer_command_string(run_dir: Path) -> str:
    """Return a human-readable viewer command string for display."""
    return " ".join(build_viewer_command(run_dir))
