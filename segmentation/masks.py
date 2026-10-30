"""Optional mask cleanup and object visualization."""

from collections import deque
import cv2
import numpy as np

COLORS = [(0, 180, 255), (80, 220, 80), (255, 140, 80), (210, 80, 210), (80, 220, 220)]


def cleanup_mask(mask: np.ndarray, max_hole_area: int = 0,
                 min_component_area: int = 0) -> np.ndarray:
    """Conservative cleanup: explicit area thresholds, no erosion or dilation."""
    if max_hole_area < 0 or min_component_area < 0:
        raise ValueError("Cleanup thresholds cannot be negative")
    result = mask.astype(np.uint8).copy()
    if min_component_area:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < min_component_area:
                result[labels == index] = 0
    if max_hole_area:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - result, connectivity=8)
        height, width = result.shape
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            touches_border = x == 0 or y == 0 or x + w == width or y + h == height
            if not touches_border and area <= max_hole_area:
                result[labels == index] = 1
    return result.astype(bool)


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def render_frame(frame, masks, trajectories, show_tracks: bool):
    overlay = frame.copy()
    union = np.zeros(frame.shape[:2], dtype=bool)
    for object_id, mask in masks.items():
        color = COLORS[(object_id - 1) % len(COLORS)]
        union |= mask
        overlay[mask] = (0.55 * frame[mask] + 0.45 * np.array(color)).astype(np.uint8)
        center = centroid(mask)
        path = trajectories.setdefault(object_id, deque(maxlen=150))
        if center is None:
            path.clear()  # Do not draw a straight line across a disappearance.
            continue
        location = tuple(round(value) for value in center)
        path.append(location)
        if show_tracks:
            cv2.circle(overlay, location, 4, color, -1)
            cv2.putText(overlay, f"ID {object_id}", (location[0] + 5, location[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if len(path) > 1:
                cv2.polylines(overlay, [np.array(path, dtype=np.int32)], False, color, 2)
    foreground = np.zeros_like(frame)
    foreground[union] = frame[union]
    return foreground, overlay
