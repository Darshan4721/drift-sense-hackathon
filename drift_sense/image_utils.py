"""Image I/O, geometry, and low-level feature utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import IMAGE_SIZE


def load_grayscale(path: str | Path, expected_size: int | None = IMAGE_SIZE) -> np.ndarray:
    """Load PNG/JPEG/TIFF as float32 grayscale in the range [0, 1]."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Expected 3-channel RGB/BGR or 4-channel RGBA/BGRA image, got shape {image.shape}")
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {image.shape}")
    if expected_size is not None and image.shape != (expected_size, expected_size):
        raise ValueError(
            f"Expected {expected_size}x{expected_size} image, got {image.shape} for {image_path}"
        )
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0 if image.max() <= 255.0 else 65535.0
    return np.clip(image, 0.0, 1.0)


def save_grayscale(path: str | Path, image: np.ndarray) -> None:
    """Save a normalized floating-point grayscale image."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    if not cv2.imwrite(str(output), np.round(array * 255.0).astype(np.uint8)):
        raise OSError(f"Unable to write image: {output}")


def normalize_contrast(image: np.ndarray) -> np.ndarray:
    """Standardize contrast robustly using percentile clipping."""

    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, [1.0, 99.0])
    if high - low < 1e-6:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute normalized Sobel gradient magnitude."""

    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    maximum = float(magnitude.max(initial=0.0))
    return magnitude / maximum if maximum > 1e-8 else magnitude


def edge_enhance(image: np.ndarray, alpha: float) -> np.ndarray:
    """Add gradient magnitude to an image for configurable edge brightening."""

    return np.clip(image + float(alpha) * gradient_magnitude(image), 0.0, 1.0)


def apply_dog(image: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.5) -> np.ndarray:
    """Difference-of-Gaussians bandpass filter to suppress low-frequency illumination and high-frequency noise."""

    g1 = cv2.GaussianBlur(image.astype(np.float32), (0, 0), float(sigma1))
    g2 = cv2.GaussianBlur(image.astype(np.float32), (0, 0), float(sigma2))
    dog = g1 - g2
    return normalize_contrast(dog)


def apply_log(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Laplacian-of-Gaussian filter for edge and feature enhancement."""

    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), float(sigma))
    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    return normalize_contrast(log)


def apply_lcn(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Local contrast normalization using local Gaussian neighborhood mean and variance."""

    img = image.astype(np.float32)
    mu = cv2.GaussianBlur(img, (0, 0), float(sigma))
    diff = img - mu
    var = cv2.GaussianBlur(diff * diff, (0, 0), float(sigma))
    std = np.sqrt(np.maximum(var, 0.0)) + 1e-4
    lcn = diff / std
    return normalize_contrast(lcn)


def apply_prefilter(image: np.ndarray, method: str = "none") -> np.ndarray:
    """Apply the specified prefilter method ('none', 'dog', 'log', 'lcn'). Defaults to 'none'."""

    if method == "none" or not method:
        return image
    if method == "dog":
        return apply_dog(image)
    if method == "log":
        return apply_log(image)
    if method == "lcn":
        return apply_lcn(image)
    raise ValueError(f"Unknown prefilter method: {method}. Allowed: 'none', 'dog', 'log', 'lcn'")


def center_from_bbox(x0: float, y0: float, width: float, height: float) -> tuple[float, float]:
    """Convert top-left plus dimensions to an (x, y) center."""

    return x0 + width / 2.0, y0 + height / 2.0


def bbox_from_center(
    center_x: float, center_y: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """Convert center and dimensions to x0, y0, x1, y1."""

    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def clamp_bbox(
    bbox: Iterable[float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Clamp a floating-point bounding box to valid integer pixel bounds."""

    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    x0 = max(0, min(image_width - 1, x0))
    y0 = max(0, min(image_height - 1, y0))
    x1 = max(x0 + 1, min(image_width, x1))
    y1 = max(y0 + 1, min(image_height, y1))
    return x0, y0, x1, y1


def safe_crop(image: np.ndarray, center_x: float, center_y: float, width: int, height: int) -> np.ndarray:
    """Extract a padded crop centered at (x, y)."""

    half_w, half_h = width // 2, height // 2
    cx, cy = int(round(center_x)), int(round(center_y))
    x0, x1 = cx - half_w, cx - half_w + width
    y0, y1 = cy - half_h, cy - half_h + height
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - image.shape[1]), max(0, y1 - image.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        x0, x1 = x0 + pad_left, x1 + pad_left
        y0, y1 = y0 + pad_top, y1 + pad_top
    return image[y0:y1, x0:x1].copy()
