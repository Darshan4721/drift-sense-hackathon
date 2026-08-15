"""Visualization utilities for localization predictions."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .image_utils import bbox_from_center


def render_result(
    reference: np.ndarray,
    search: np.ndarray,
    prediction: tuple[float, float],
    truth: tuple[float, float] | None = None,
    predicted_size: tuple[float, float] = (100.0, 100.0),
    truth_bbox: tuple[float, float, float, float] | None = None,
    confidence: float | None = None,
    output: str | Path | None = None,
    title: str = "Drift-Sense localization",
) -> None:
    """Render a readable diagnostic image."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].imshow(reference, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Reference (100x magnification)")
    axes[0].axis("off")
    axes[1].imshow(search, cmap="gray", vmin=0, vmax=1)
    px, py = prediction
    width, height = predicted_size
    x0, y0, x1, y1 = bbox_from_center(px, py, width, height)
    axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", linewidth=1.8, label="prediction"))
    axes[1].scatter([px], [py], c="cyan", s=40)
    text_lines = [f"predicted: ({px:.1f}, {py:.1f})"]
    if truth is not None:
        tx, ty = truth
        axes[1].scatter([tx], [ty], c="lime", marker="x", s=55, linewidths=2, label="truth")
        if truth_bbox is not None:
            bx0, by0, bx1, by1 = truth_bbox
            axes[1].add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, color="lime", linewidth=1.2, linestyle="--"))
        error = float(np.hypot(px - tx, py - ty))
        text_lines.append(f"truth: ({tx:.1f}, {ty:.1f})")
        text_lines.append(f"error: {error:.2f}px")
    if confidence is not None:
        text_lines.append(f"confidence: {confidence:.3f}")
    axes[1].set_title("Search image\n" + " | ".join(text_lines))
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axis("off")
    fig.suptitle(title)
    if output is None:
        plt.show()
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
