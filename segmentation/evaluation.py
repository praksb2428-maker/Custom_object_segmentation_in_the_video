"""Evaluate binary masks without treating missing predictions as empty masks."""

import csv
from pathlib import Path
import numpy as np
import cv2
from .video import read_image


def scores(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and ground-truth dimensions differ")
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    total = int(prediction.sum()) + int(truth.sum())
    return (intersection / union if union else 1.0,
            2 * intersection / total if total else 1.0)


def binary_mask(path: Path) -> np.ndarray:
    image = read_image(path, cv2.IMREAD_UNCHANGED)
    if image.ndim != 2 or not np.isin(image, [0, 255]).all():
        raise ValueError(f"Ground truth / prediction must be a single-channel 0/255 PNG: {path}")
    return image > 0


def evaluate(predictions: Path, truth_root: Path, csv_path: Path) -> dict:
    if csv_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {csv_path}")
    if not truth_root.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {truth_root}")
    rows = []
    for object_dir in sorted(truth_root.glob("object_*")):
        if not object_dir.is_dir():
            continue
        object_id = int(object_dir.name.split("_")[1])
        for truth_path in sorted(object_dir.glob("*.png")):
            prediction_path = predictions / object_dir.name / truth_path.name
            if not prediction_path.is_file():
                raise FileNotFoundError(f"Missing prediction for ground truth: {prediction_path}")
            truth, prediction = binary_mask(truth_path), binary_mask(prediction_path)
            iou, dice = scores(prediction, truth)
            rows.append([object_id, truth_path.stem, int(truth.sum()), int(prediction.sum()), iou, dice])
    if not rows:
        raise ValueError("No ground-truth PNGs found under object_<id> directories")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object_id", "frame_index", "truth_area", "prediction_area", "iou", "dice"])
        writer.writerows(rows)
    visible = [row for row in rows if row[2] > 0]
    return {"evaluated_pairs": len(rows), "visible_pairs": len(visible),
            "mean_iou": float(np.mean([row[4] for row in rows])),
            "mean_dice": float(np.mean([row[5] for row in rows])),
            "visible_mean_iou": float(np.mean([row[4] for row in visible])) if visible else None,
            "empty_both_score": 1.0}
