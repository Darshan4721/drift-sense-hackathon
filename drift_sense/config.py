"""Shared configuration and reproducibility helpers for Drift-Sense."""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SIZE = 1000
DEFAULT_SCALE = 0.1
SEARCH_SCALES = (0.095, 0.098, 0.100, 0.102, 0.105)


@dataclass(frozen=True)
class AugmentationConfig:
    """Configurable image degradation parameters."""

    gaussian_sigma_reference: float = 2.0
    gaussian_sigma_search: float = 5.0
    poisson_strength_reference: float = 20.0
    poisson_strength_search: float = 10.0
    blur_sigma_reference: float = 0.6
    blur_sigma_search: float = 1.2
    max_rotation_degrees: float = 1.5
    min_scale: float = 0.985
    max_scale: float = 1.015
    max_translation: float = 3.0
    contrast_min: float = 0.90
    contrast_max: float = 1.10
    illumination_sigma: float = 0.08
    edge_alpha_min: float = 0.0
    edge_alpha_max: float = 0.08
    impulse_probability: float = 0.0004


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and optional Torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment_info() -> dict[str, Any]:
    """Return a JSON-serializable environment summary."""

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
    return info


def save_json(path: str | Path, payload: Any) -> None:
    """Write formatted UTF-8 JSON, creating the parent directory."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass to a plain dictionary."""

    return asdict(value)
