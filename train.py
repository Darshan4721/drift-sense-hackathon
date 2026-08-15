#!/usr/bin/env python3
"""Train the optional lightweight candidate verifier."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from drift_sense.classical import generate_candidates, rerank_candidates
from drift_sense.config import environment_info, save_json, seed_everything
from drift_sense.image_utils import load_grayscale
from drift_sense.model import PairDataset, TinyVerifier, prepare_pair


def _split_ids(data_dir: Path, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Create disjoint ID-level train, validation, and test partitions."""

    with (data_dir / "annotations.csv").open(encoding="utf-8", newline="") as handle:
        ids = sorted(int(row["sample_id"]) for row in csv.DictReader(handle))
    if len(ids) < 3:
        raise ValueError("At least three annotated samples are required for train/validation/test splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    train_end = min(n - 2, max(1, int(0.70 * n)))
    val_end = min(n - 1, max(train_end + 1, int(0.85 * n)))
    return ids[:train_end], ids[train_end:val_end], ids[val_end:]


def _validation_localization_error(
    model: Any,
    data_dir: Path,
    sample_ids: list[int],
    rows: dict[int, dict[str, str]],
    device: Any,
) -> float:
    """Measure coordinate error after classical candidate generation and learned reranking."""

    import torch

    errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for sample_id in sample_ids:
            row = rows[sample_id]
            reference = load_grayscale(data_dir / row["reference_path"])
            search = load_grayscale(data_dir / row["search_path"])
            candidates = generate_candidates(reference, search, top_k_per_scale=4)
            if not candidates:
                continue
            scores: dict[int, float] = {}
            for index, candidate in enumerate(candidates):
                pair = prepare_pair(reference, search, candidate.x, candidate.y)
                logits = model(torch.from_numpy(pair[None, ...]).to(device))
                scores[index] = float(torch.sigmoid(logits)[0].item() * 2.0 - 1.0)
            best = rerank_candidates(candidates, search.shape, scores)[0]
            errors.append(float(np.hypot(best.x - float(row["x"]), best.y - float(row["y"]))))
    return float(np.mean(errors)) if errors else math.inf


def train(data_dir: str | Path, output_dir: str | Path, epochs: int, batch_size: int, seed: int, patience: int = 7) -> dict[str, Any]:
    """Train and save best.pt; returns final metrics."""

    try:
        import torch
        from torch import nn
        from torch.amp import GradScaler, autocast
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Training requires PyTorch. Install the pinned requirements or use classical inference.") from exc

    seed_everything(seed)
    data_path, output_path = Path(data_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, test_ids = _split_ids(data_path, seed)
    rows: dict[int, dict[str, str]] = {}
    with (data_path / "annotations.csv").open(encoding="utf-8", newline="") as handle:
        rows = {int(row["sample_id"]): row for row in csv.DictReader(handle)}
    train_set = PairDataset(data_path, train_ids, seed=seed, samples_per_id=4)
    val_set = PairDataset(data_path, val_ids, seed=seed + 1, samples_per_id=4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_metric = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        for pairs, labels in train_loader:
            pairs, labels = pairs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(pairs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for pairs, labels in val_loader:
                pairs, labels = pairs.to(device), labels.to(device)
                logits = model(pairs)
                loss = criterion(logits, labels)
                val_losses.append(float(loss.detach().cpu()))
                correct += int(((torch.sigmoid(logits) >= 0.5) == (labels >= 0.5)).sum().item())
                total += int(labels.numel())
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        val_coordinate_error = _validation_localization_error(model, data_path, val_ids, rows, device)
        scheduler.step(val_coordinate_error)
        record = {
            "epoch": epoch,
            "train_pair_loss": train_loss,
            "val_pair_loss": val_loss,
            "val_pair_accuracy": correct / max(1, total),
            "val_localization_error": val_coordinate_error,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if val_coordinate_error < best_metric:
            best_metric, best_epoch, stale_epochs = val_coordinate_error, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_localization_error": val_coordinate_error,
                    "val_pair_loss": val_loss,
                    "seed": seed,
                    "architecture": "TinyVerifier",
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                },
                output_path / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    with (output_path / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    final = {
        "best_validation_localization_error": best_metric,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "test_count": len(test_ids),
        "device": str(device),
        "environment": environment_info(),
    }
    save_json(output_path / "final_metrics.json", final)
    save_json(output_path / "split.json", {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids, "seed": seed})
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.seed, args.patience)


if __name__ == "__main__":
    main()
