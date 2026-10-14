"""FaceOrd: landmark-guided ordinal multi-task learning for facial age and gender prediction.

The model classes are imported lazily so that evaluation utilities work without PyTorch.
"""
from .config import Config

__version__ = "1.1.0"
__all__ = ["Config", "FaceOrd", "build_faceord"]


def __getattr__(name):
    if name in {"FaceOrd", "build_faceord"}:
        from .models import FaceOrd, build_faceord
        return {"FaceOrd": FaceOrd, "build_faceord": build_faceord}[name]
    raise AttributeError(name)
