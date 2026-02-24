"""core/tasks/__init__.py"""
from .nuclei import run_nuclei_detection
from .semantic import run_semantic_segmentation
from .patch_pred import run_patch_prediction

TASK_MAP = {
    "nuclei_detection": run_nuclei_detection,
    "semantic_segmentation": run_semantic_segmentation,
    "patch_prediction": run_patch_prediction,
}

__all__ = [
    "run_nuclei_detection",
    "run_semantic_segmentation",
    "run_patch_prediction",
    "TASK_MAP",
]
