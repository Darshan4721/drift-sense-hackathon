from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from drift_sense.image_utils import bbox_from_center, center_from_bbox, load_grayscale, save_grayscale
from drift_sense.synthetic import generate_pair, write_dataset


def test_coordinate_conversion_round_trip() -> None:
    center = center_from_bbox(10.0, 20.0, 100.0, 80.0)
    assert center == (60.0, 60.0)
    bbox = bbox_from_center(*center, 100.0, 80.0)
    assert bbox == (10.0, 20.0, 110.0, 100.0)


def test_generated_pair_has_consistent_geometry() -> None:
    reference, search, metadata = generate_pair(1, "DRAM", 43)
    assert reference.shape == (1000, 1000)
    assert search.shape == (1000, 1000)
    bbox = metadata["target_bbox"]
    assert bbox["x0"] < metadata["target_center_x"] < bbox["x1"]
    assert bbox["y0"] < metadata["target_center_y"] < bbox["y1"]
    assert 0 < metadata["target_center_x"] < 1000
    assert 0 < metadata["target_center_y"] < 1000
    assert metadata["scale_factor"] == 0.1
    assert metadata["reference_to_search_scale"] == 0.1
    assert metadata["target_bbox"]["x0"] <= metadata["target_center_x"] <= metadata["target_bbox"]["x1"]
    assert metadata["target_bbox"]["y0"] <= metadata["target_center_y"] <= metadata["target_bbox"]["y1"]


def test_dataset_annotation_and_image_loading(tmp_path: Path) -> None:
    write_dataset(tmp_path, 1, "DRAM", 42)
    with (tmp_path / "annotations.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    reference = load_grayscale(tmp_path / row["reference_path"])
    search = load_grayscale(tmp_path / row["search_path"])
    assert reference.shape == search.shape == (1000, 1000)
    assert 0 <= float(row["x"]) <= 1000
    assert 0 <= float(row["y"]) <= 1000


def test_image_loader_converts_rgb(tmp_path: Path) -> None:
    import cv2

    rgb_path = tmp_path / "rgb.png"
    rgb = np.zeros((1000, 1000, 3), dtype=np.uint8)
    rgb[:, :, 1] = 255
    assert cv2.imwrite(str(rgb_path), rgb)
    loaded = load_grayscale(rgb_path)
    assert loaded.shape == (1000, 1000)
    assert loaded.dtype == np.float32
    rgba_path = tmp_path / "rgba.png"
    rgba = np.zeros((1000, 1000, 4), dtype=np.uint8)
    rgba[:, :, 0] = 20
    rgba[:, :, 1] = 100
    rgba[:, :, 2] = 200
    rgba[:, :, 3] = 128
    assert cv2.imwrite(str(rgba_path), rgba)
    rgba_loaded = load_grayscale(rgba_path)
    assert rgba_loaded.shape == (1000, 1000)
    assert rgba_loaded.dtype == np.float32


def test_inference_output_format(tmp_path: Path) -> None:
    output = tmp_path / "data"
    write_dataset(output, 1, "DRAM", 42)
    reference = output / "reference" / "ref_000001.png"
    search = output / "search" / "search_000001.png"
    command = [sys.executable, "localize.py", "--reference", str(reference), "--search", str(search), "--method", "classical"]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1])
    assert completed.stderr == ""
    assert completed.stdout.strip().startswith("(")
    assert completed.stdout.strip().endswith(")")


def test_malformed_checkpoint_falls_back_to_classical(tmp_path: Path) -> None:
    output = tmp_path / "data"
    write_dataset(output, 1, "DRAM", 42)
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    command = [
        sys.executable,
        "localize.py",
        "--reference",
        str(output / "reference" / "ref_000001.png"),
        "--search",
        str(output / "search" / "search_000001.png"),
        "--method",
        "hybrid",
        "--checkpoint",
        str(checkpoint),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1])
    assert completed.stderr == ""
    assert completed.stdout.strip().startswith("(")


def test_save_grayscale_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    source = np.full((1000, 1000), 0.5, dtype=np.float32)
    save_grayscale(path, source)
    loaded = load_grayscale(path)
    assert loaded.shape == source.shape
    assert abs(float(loaded.mean()) - 0.5) < 0.01
