"""Compact neural verifier for candidate pairs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - classical-only installations remain supported.
    torch = None
    nn = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

from .image_utils import load_grayscale, safe_crop

PATCH_SIZE = 96


def prepare_pair(reference: np.ndarray, search: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
    """Resize reference and crop a search candidate to a two-channel tensor."""

    reference_small = cv2.resize(reference, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    candidate = safe_crop(search, center_x, center_y, PATCH_SIZE, PATCH_SIZE)
    candidate = cv2.resize(candidate, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    reference_small = (reference_small - reference_small.mean()) / (reference_small.std() + 1e-6)
    candidate = (candidate - candidate.mean()) / (candidate.std() + 1e-6)
    return np.stack([reference_small, candidate], axis=0).astype(np.float32)


if torch is not None:

    class TinyVerifier(nn.Module):
        """Small two-branch-compatible CNN that scores a reference/candidate pair."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(2, 16, 5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(32, 1))

        def forward(self, pair: torch.Tensor) -> torch.Tensor:
            return self.head(self.features(pair)).squeeze(1)

else:

    class TinyVerifier:  # type: ignore[no-redef]
        """Placeholder that gives a clear error when deep mode lacks PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for the learned verifier")


class PairDataset(Dataset):
    """Generate positive and hard-ish negative candidate pairs from annotations."""

    def __init__(self, data_dir: str | Path, sample_ids: list[int], seed: int = 42, samples_per_id: int = 4) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for training")
        self.data_dir = Path(data_dir)
        self.rows: dict[int, dict[str, str]] = {}
        with (self.data_dir / "annotations.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.rows[int(row["sample_id"])] = row
        self.items = [(sample_id, offset) for sample_id in sample_ids for offset in range(samples_per_id)]
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id, offset = self.items[index]
        row = self.rows[sample_id]
        reference = load_grayscale(self.data_dir / row["reference_path"])
        search = load_grayscale(self.data_dir / row["search_path"])
        truth_x, truth_y = float(row["x"]), float(row["y"])
        rng = np.random.default_rng(self.seed + sample_id * 1009 + offset)
        if offset % 2 == 0:
            x, y, label = truth_x + float(rng.normal(0, 2.0)), truth_y + float(rng.normal(0, 2.0)), 1.0
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radius = float(rng.uniform(45.0, 300.0))
            x, y, label = truth_x + np.cos(angle) * radius, truth_y + np.sin(angle) * radius, 0.0
        pair = prepare_pair(reference, search, x, y)
        return torch.from_numpy(pair), torch.tensor(label, dtype=torch.float32)


def load_verifier(checkpoint: str | Path, device: str = "cpu") -> tuple[Any, dict[str, Any]] | tuple[None, dict[str, Any]]:
    """Load a verifier checkpoint defensively; invalid or absent weights are ignored."""

    if torch is None:
        return None, {"reason": "torch_unavailable"}
    path = Path(checkpoint)
    if not path.is_file():
        return None, {"reason": "checkpoint_missing", "path": str(path)}
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TinyVerifier()
        model.load_state_dict(payload["model_state"] if "model_state" in payload else payload)
        model.to(device)
        model.eval()
        safe_details: dict[str, Any] = {"path": str(path)}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key == "model_state":
                    continue
                if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                    safe_details[key] = value
        return model, safe_details
    except Exception as exc:  # A checkpoint is optional; every load failure must fall back safely.
        return None, {"reason": "checkpoint_invalid", "error": f"{type(exc).__name__}: {exc}"}
