"""Synthetic semiconductor-like layout generation and image degradation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AugmentationConfig, IMAGE_SIZE, dataclass_to_dict, save_json
from .image_utils import edge_enhance, save_grayscale


def _draw_dram_layout(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a continuous DRAM-inspired word-line/bit-line layout."""

    canvas = np.full((size, size), 0.16, dtype=np.float32)
    scale = max(0.5, size / 1000.0)
    period_x = int(rng.integers(max(8, round(18 * scale)), max(10, round(26 * scale))))
    period_y = int(rng.integers(max(9, round(20 * scale)), max(11, round(30 * scale))))
    bitline_offset = int(rng.integers(0, period_x))
    wordline_offset = int(rng.integers(0, period_y))

    # Smooth field illumination makes the synthetic scene less perfectly uniform.
    yy, xx = np.mgrid[0:size, 0:size]
    field = 0.025 * np.sin(xx / rng.uniform(250.0, 400.0))
    field += 0.020 * np.cos(yy / rng.uniform(300.0, 500.0))
    field += 0.012 * (xx / size) * (yy / size)
    canvas += field.astype(np.float32)

    for x in range(bitline_offset, size, period_x):
        width = int(rng.integers(3, 7))
        intensity = float(rng.uniform(0.42, 0.72))
        cv2.line(canvas, (x, 0), (x, size - 1), intensity, width, cv2.LINE_AA)
    for y in range(wordline_offset, size, period_y):
        width = int(rng.integers(3, 7))
        intensity = float(rng.uniform(0.55, 0.92))
        cv2.line(canvas, (0, y), (size - 1, y), intensity, width, cv2.LINE_AA)

    for y in range(wordline_offset, size, period_y):
        for x in range(bitline_offset, size, period_x):
            radius = int(rng.integers(4, 9))
            contact_intensity = float(rng.uniform(0.68, 1.0))
            if rng.random() > 0.035:
                cv2.circle(canvas, (x, y), radius, contact_intensity, -1, cv2.LINE_AA)
            else:
                # A weak contact is preferable to a completely unrelated defect.
                cv2.circle(canvas, (x, y), max(2, radius - 2), contact_intensity * 0.65, -1, cv2.LINE_AA)

    # Small process-like local structures preserve the periodic architecture while
    # preventing every tile from being identical.
    for _ in range(max(20, size // 120)):
        x0 = int(rng.integers(0, size))
        y0 = int(rng.integers(0, size))
        length = int(rng.integers(15, 60))
        cv2.line(
            canvas,
            (x0, y0),
            (min(size - 1, x0 + length), y0 + int(rng.integers(-2, 3))),
            float(rng.uniform(0.10, 0.30)),
            int(rng.integers(1, 3)),
            cv2.LINE_AA,
        )
    return np.clip(canvas, 0.0, 1.0)


def _draw_finfet_layout(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create an optional FinFET-inspired continuous layout."""

    canvas = np.full((size, size), 0.14, dtype=np.float32)
    fin_period = int(rng.integers(34, 48))
    gate_period = int(rng.integers(160, 240))
    for x in range(int(rng.integers(0, fin_period)), size, fin_period):
        cv2.line(
            canvas,
            (x, 0),
            (x + int(rng.integers(-5, 6)), size - 1),
            float(rng.uniform(0.40, 0.75)),
            int(rng.integers(3, 7)),
            cv2.LINE_AA,
        )
    for y in range(int(rng.integers(0, gate_period)), size, gate_period):
        cv2.rectangle(
            canvas,
            (0, y),
            (size - 1, min(size - 1, y + int(rng.integers(8, 17)))),
            float(rng.uniform(0.68, 0.95)),
            -1,
        )
    for _ in range(max(30, size // 100)):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        cv2.circle(canvas, (x, y), int(rng.integers(3, 9)), float(rng.uniform(0.65, 1.0)), -1)
    return np.clip(canvas, 0.0, 1.0)


def make_continuous_layout(size: int, architecture: str, seed: int) -> np.ndarray:
    """Generate a larger continuous layout from which both views are sampled."""

    rng = np.random.default_rng(seed)
    architecture_upper = architecture.upper()
    if architecture_upper == "DRAM":
        return _draw_dram_layout(size, rng)
    if architecture_upper == "FINFET":
        return _draw_finfet_layout(size, rng)
    raise ValueError(f"Unsupported architecture {architecture!r}; use DRAM or FinFET")


def _warp_image(image: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply an affine transform and return the image plus its affine matrix."""

    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return transformed, matrix


def _affine_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return (
        float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
        float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
    )


def degrade_image(
    image: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
    view: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply independent sensor, optical, illumination, and geometric effects."""

    if view not in {"reference", "search"}:
        raise ValueError("view must be 'reference' or 'search'")
    result = np.asarray(image, dtype=np.float32).copy()
    stronger = view == "search"
    gaussian_sigma = config.gaussian_sigma_search if stronger else config.gaussian_sigma_reference
    blur_sigma = config.blur_sigma_search if stronger else config.blur_sigma_reference
    poisson_strength = config.poisson_strength_search if stronger else config.poisson_strength_reference

    contrast = float(rng.uniform(config.contrast_min, config.contrast_max))
    result = np.clip((result - 0.5) * contrast + 0.5, 0.0, 1.0)
    illumination = rng.normal(0.0, config.illumination_sigma, size=(max(4, result.shape[0] // 30), max(4, result.shape[1] // 30))).astype(np.float32)
    illumination = cv2.resize(illumination, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
    result = np.clip(result + illumination, 0.0, 1.0)
    if blur_sigma > 0:
        result = cv2.GaussianBlur(result, (0, 0), blur_sigma)
    if gaussian_sigma > 0:
        result += rng.normal(0.0, gaussian_sigma / 255.0, result.shape).astype(np.float32)
    # Poisson-like shot noise is sampled independently for each view.
    scale = max(1.0, poisson_strength)
    result += (rng.poisson(np.clip(result, 0.0, 1.0) * scale) / scale - np.clip(result, 0.0, 1.0)).astype(np.float32)
    edge_alpha = float(rng.uniform(config.edge_alpha_min, config.edge_alpha_max))
    result = edge_enhance(result, edge_alpha)
    impulse_mask = rng.random(result.shape) < config.impulse_probability
    if np.any(impulse_mask):
        result[impulse_mask] = rng.choice(np.array([0.0, 1.0], dtype=np.float32), size=int(impulse_mask.sum()))

    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    scale_factor = float(rng.uniform(config.min_scale, config.max_scale))
    tx = float(rng.uniform(-config.max_translation, config.max_translation))
    ty = float(rng.uniform(-config.max_translation, config.max_translation))
    result, _ = _warp_image(result, angle, scale_factor, tx, ty)
    return np.clip(result, 0.0, 1.0), {
        "gaussian_sigma": gaussian_sigma,
        "blur_sigma": blur_sigma,
        "poisson_strength": poisson_strength,
        "contrast": contrast,
        "edge_alpha": edge_alpha,
        "rotation_degrees": angle,
        "scale_multiplier": scale_factor,
        "translation_x": tx,
        "translation_y": ty,
    }


def generate_pair(
    sample_id: int,
    architecture: str,
    seed: int,
    image_size: int = IMAGE_SIZE,
    config: AugmentationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one paired reference/search sample with exact target geometry."""

    config = config or AugmentationConfig()
    # Native layout dimensions are deliberately compact: a 200-pixel native
    # reference is upsampled to 1000 pixels, while a 2000-pixel native search
    # field is downsampled to 1000 pixels. This preserves the 10x relation
    # without allocating a 15000x15000 float array.
    native_reference = max(160, image_size // 5)
    search_width = native_reference * 10
    layout_size = search_width + native_reference * 2
    layout = make_continuous_layout(layout_size, architecture, seed)
    rng = np.random.default_rng(seed)
    max_search_origin = layout_size - search_width
    search_origin_x = int(rng.integers(0, max_search_origin + 1))
    search_origin_y = int(rng.integers(0, max_search_origin + 1))
    target_margin = native_reference * 2
    ref_origin_x = int(rng.integers(search_origin_x + target_margin, search_origin_x + search_width - native_reference - target_margin + 1))
    ref_origin_y = int(rng.integers(search_origin_y + target_margin, search_origin_y + search_width - native_reference - target_margin + 1))
    # Add a connected, sample-specific process signature within the continuous
    # layout. It is not a pasted reference: it consists of plausible local
    # contact/line variations and is therefore present in both magnifications.
    signature_rng = np.random.default_rng(seed + 97)
    for _ in range(7):
        sx = int(rng.integers(ref_origin_x + 10, ref_origin_x + native_reference - 10))
        sy = int(rng.integers(ref_origin_y + 10, ref_origin_y + native_reference - 10))
        radius = int(signature_rng.integers(3, 8))
        cv2.circle(layout, (sx, sy), radius, float(signature_rng.uniform(0.22, 0.95)), -1, cv2.LINE_AA)
    for _ in range(3):
        sy = int(rng.integers(ref_origin_y + 8, ref_origin_y + native_reference - 8))
        cv2.line(layout, (ref_origin_x + 6, sy), (ref_origin_x + native_reference - 6, sy + int(signature_rng.integers(-2, 3))), float(signature_rng.uniform(0.18, 0.85)), int(signature_rng.integers(1, 3)), cv2.LINE_AA)
    for _ in range(5):
        sx = int(rng.integers(ref_origin_x + 12, ref_origin_x + native_reference - 32))
        sy = int(rng.integers(ref_origin_y + 12, ref_origin_y + native_reference - 32))
        side = int(signature_rng.integers(12, 28))
        cv2.rectangle(layout, (sx, sy), (sx + side, sy + side), float(signature_rng.uniform(0.10, 0.98)), -1)

    reference_base = layout[ref_origin_y : ref_origin_y + native_reference, ref_origin_x : ref_origin_x + native_reference]
    search_base = layout[search_origin_y : search_origin_y + search_width, search_origin_x : search_origin_x + search_width]
    reference_native = cv2.resize(reference_base, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    reference, reference_params = degrade_image(reference_native, np.random.default_rng(seed + 1), config, "reference")
    search_downsampled = cv2.resize(search_base, (image_size, image_size), interpolation=cv2.INTER_AREA)
    search, search_params = degrade_image(search_downsampled, np.random.default_rng(seed + 2), config, "search")

    native_to_search = image_size / float(search_width)
    raw_x = (ref_origin_x - search_origin_x + native_reference / 2.0) * native_to_search
    raw_y = (ref_origin_y - search_origin_y + native_reference / 2.0) * native_to_search
    # The search affine transform changes the target center; account for it.
    _, search_matrix = _warp_image(search_downsampled, search_params["rotation_degrees"], search_params["scale_multiplier"], search_params["translation_x"], search_params["translation_y"])
    target_x, target_y = _affine_point(search_matrix, raw_x, raw_y)
    half_native_side = native_reference * native_to_search / 2.0
    raw_corners = [(raw_x - half_native_side, raw_y - half_native_side), (raw_x + half_native_side, raw_y - half_native_side), (raw_x + half_native_side, raw_y + half_native_side), (raw_x - half_native_side, raw_y + half_native_side)]
    transformed_corners = [_affine_point(search_matrix, corner_x, corner_y) for corner_x, corner_y in raw_corners]
    bbox = {
        "x0": max(0.0, min(point[0] for point in transformed_corners)),
        "y0": max(0.0, min(point[1] for point in transformed_corners)),
        "x1": min(float(image_size), max(point[0] for point in transformed_corners)),
        "y1": min(float(image_size), max(point[1] for point in transformed_corners)),
    }
    target_x = float(np.clip(target_x, 1.0, image_size - 1.0))
    target_y = float(np.clip(target_y, 1.0, image_size - 1.0))
    metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "architecture": architecture.upper(),
        "seed": seed,
        "reference_size": [image_size, image_size],
        "search_size": [image_size, image_size],
        "native_reference_size": [native_reference, native_reference],
        "native_search_size": [search_width, search_width],
        "native_to_final_search_scale": native_to_search,
        "scale_factor": native_reference * native_to_search / float(image_size),
        "reference_to_search_scale": native_reference * native_to_search / float(image_size),
        "target_center_x": target_x,
        "target_center_y": target_y,
        "target_bbox": bbox,
        "reference_origin_layout": [ref_origin_x, ref_origin_y],
        "search_origin_layout": [search_origin_x, search_origin_y],
        "reference_augmentation": reference_params,
        "search_augmentation": search_params,
        "augmentation_config": dataclass_to_dict(config),
    }
    return reference, search, metadata


def write_dataset(
    output_dir: str | Path,
    num_pairs: int,
    architecture: str = "DRAM",
    seed: int = 42,
    config: AugmentationConfig | None = None,
    save_visualizations: bool = False,
) -> dict[str, Any]:
    """Generate images, annotations.csv, metadata.json, and optional previews."""

    output = Path(output_dir)
    reference_dir, search_dir = output / "reference", output / "search"
    reference_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for index in range(1, num_pairs + 1):
        reference, search, item = generate_pair(index, architecture, seed + index, config=config)
        ref_path = reference_dir / f"ref_{index:06d}.png"
        search_path = search_dir / f"search_{index:06d}.png"
        save_grayscale(ref_path, reference)
        save_grayscale(search_path, search)
        bbox = item["target_bbox"]
        row = {
            "sample_id": index,
            "reference_path": str(ref_path.relative_to(output)),
            "search_path": str(search_path.relative_to(output)),
            "x": item["target_center_x"],
            "y": item["target_center_y"],
            "bbox_x0": bbox["x0"],
            "bbox_y0": bbox["y0"],
            "bbox_x1": bbox["x1"],
            "bbox_y1": bbox["y1"],
            "architecture": item["architecture"],
            "scale_factor": item["scale_factor"],
            "reference_to_search_scale": item["reference_to_search_scale"],
            "rotation": item["search_augmentation"]["rotation_degrees"],
            "blur_sigma": item["search_augmentation"]["blur_sigma"],
            "noise_sigma": item["search_augmentation"]["gaussian_sigma"],
            "seed": item["seed"],
        }
        rows.append(row)
        metadata.append(item)
        if save_visualizations and index <= 12:
            preview = np.zeros((IMAGE_SIZE, IMAGE_SIZE * 2), dtype=np.float32)
            preview[:, :IMAGE_SIZE] = reference
            preview[:, IMAGE_SIZE:] = search
            save_grayscale(output / "previews" / f"pair_{index:06d}.png", preview)

    import csv

    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    from .config import environment_info

    save_json(output / "metadata.json", {"architecture": architecture.upper(), "num_pairs": num_pairs, "seed": seed, "samples": metadata})
    save_json(output / "environment.json", {"generator": "Drift-Sense", "config": asdict(config or AugmentationConfig()), "environment": environment_info()})
    return {"output_dir": str(output), "num_pairs": num_pairs, "architecture": architecture.upper()}
