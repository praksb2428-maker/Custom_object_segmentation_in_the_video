"""Moving SAM 2 crops, split-mask preservation and bounded recovery."""

import csv
from functools import lru_cache
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .prompts import ObjectPrompt
from .motion import motion_support
from .video import read_image, write_image


def prompt_center(prompt):
    positive = [p for p, label in zip(prompt.points, prompt.labels) if label == 1]
    if positive:
        return tuple(np.mean(positive, axis=0))
    x1, y1, x2, y2 = prompt.box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def crop_bounds(center, side, width, height):
    cw, ch = min(side, width), min(side, height)
    x = int(np.clip(round(center[0] - cw / 2), 0, width - cw))
    y = int(np.clip(round(center[1] - ch / 2), 0, height - ch))
    return x, y, x + cw, y + ch


def local_prompt(prompt, bounds):
    x, y, right, bottom = bounds
    box = None
    if prompt.box is not None:
        x1, y1, x2, y2 = prompt.box
        box = [max(x1, x) - x, max(y1, y) - y,
               min(x2, right) - x, min(y2, bottom) - y]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Selected box is outside the tracking crop")
    points, labels = [], []
    for (px, py), label in zip(prompt.points, prompt.labels):
        if x <= px < right and y <= py < bottom:
            points.append([px - x, py - y]); labels.append(label)
        elif label == 1:
            raise ValueError("Foreground points must fit inside one crop; select each object separately")
    result = ObjectPrompt(prompt.object_id, box, points, labels)
    result.validate(right - x, bottom - y)
    return result


def target_component(mask, anchor, previous_area=None, max_step=None, reference_area=None):
    """Choose the component containing the anchor, else the nearest one.

    No minimum area: even a one-pixel target survives. This is an identity
    heuristic, not a confidence score or proof of correct segmentation.
    """
    mask = np.asarray(mask, dtype=bool)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count == 1:
        return np.zeros_like(mask)
    ax, ay = anchor
    ix, iy = int(round(ax)), int(round(ay))
    selected = int(labels[iy, ix]) if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0] else 0
    if selected == 0:
        distances = np.linalg.norm(centers[1:] - np.array(anchor), axis=1)
        selected = int(np.argmin(distances)) + 1
    return plausible_mask(labels == selected, anchor, previous_area, max_step, reference_area)


def plausible_mask(mask, anchor, previous_area=None, max_step=None, reference_area=None):
    area = int(mask.sum())
    if not area:
        return np.zeros_like(mask)
    if previous_area and (area > previous_area * 8 or area < previous_area / 8):
        return np.zeros_like(mask)
    # A peak accepted area catches gradual drift to a background fleck while
    # allowing smooth enlargement (e.g. camera zoom), unlike a fixed first area.
    if reference_area and (area < reference_area * .2 or area > reference_area * 8):
        return np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if max_step is not None and np.linalg.norm(np.array([xs.mean(), ys.mean()]) - np.array(anchor)) > max_step:
        return np.zeros_like(mask)
    return mask


def target_regions(mask, anchor, previous=None, previous_area=None, max_step=None, reference_area=None):
    """Keep SAM pixels supported by the last good target, not just one piece.

    Dilation is only a matching tolerance; it never fills output pixels.
    Distant components and broad background components remain excluded.
    """
    mask = np.asarray(mask, dtype=bool)
    if previous is None or not previous.any():
        return target_component(mask, anchor, previous_area, max_step, reference_area)
    if previous.shape != mask.shape:
        raise ValueError("Previous target dimensions do not match the crop")
    ys, xs = np.nonzero(previous)
    extent = max(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
    radius = max(8, min(64, round(extent * .1)))
    # Pixels outside this padded bounding box cannot belong to the dilated
    # support. Crop only the calculation, not the model input or output mask.
    left = max(0, int(xs.min()) - radius)
    top = max(0, int(ys.min()) - radius)
    right = min(mask.shape[1], int(xs.max()) + radius + 1)
    bottom = min(mask.shape[0], int(ys.max()) + radius + 1)
    support = cv2.dilate(previous[top:bottom, left:right].astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    overlap = np.bincount(labels[top:bottom, left:right][support].ravel(), minlength=count)
    accepted = [i for i in range(1, count) if overlap[i] >= stats[i, cv2.CC_STAT_AREA] * .25]
    selected = np.isin(labels, accepted) if accepted else np.zeros_like(mask)
    return plausible_mask(selected, anchor, previous_area, max_step, reference_area)


def mask_prompt(object_id, mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot prompt from an empty target")
    # Seed separate body parts as well, without drawing bridges between them.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    components = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:8]
    points = []
    for component in components:
        # Keep the original full-image transform: changing its image width can
        # perturb float tie-breaking and move a SAM prompt on real masks.
        padded = np.pad((labels == component).astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        py, px = np.unravel_index(np.argmax(distance), distance.shape)
        points.append([float(px), float(py)])
    return ObjectPrompt(object_id, [float(xs.min()), float(ys.min()),
                                   float(xs.max() + 1), float(ys.max() + 1)],
                        points, [1] * len(points))


class FocusedTracker:
    """Recenter every short segment or sooner at a crop boundary.

    Objects are inferred independently, then merged by frame with unchanged IDs.
    Missing frames are empty, but SAM is allowed a bounded recovery window.
    After sustained loss, the remaining output stays empty. Temporary files
    are cleaned even after failure. Recovery is a heuristic, not identity proof.
    """
    tracking_mode = "moving-object-crops"

    def __init__(self, tracker, crop_size=256, segment_frames=8, recovery_frames=12):
        if not 64 <= crop_size <= 2048 or not 1 <= segment_frames <= 32:
            raise ValueError("crop_size must be 64..2048; segment_frames must be 1..32")
        self.tracker = tracker
        self.crop_size, self.segment_frames = crop_size, segment_frames
        if type(recovery_frames) is not int or not 1 <= recovery_frames <= 60:
            raise ValueError("recovery_frames must be an integer in 1..60")
        self.recovery_frames = recovery_frames
        for name in ("model", "device", "checkpoint"):
            if hasattr(tracker, name):
                setattr(self, name, getattr(tracker, name))
        self.lost_object_ids = []
        self.missing_object_ids = []

    def predict(self, frames: Path, prompts):
        files = sorted(frames.glob("*.jpg"))
        if not files:
            raise ValueError("No input frames")
        height, width = read_image(files[0]).shape[:2]
        self.lost_object_ids = []
        self.missing_object_ids = []
        with tempfile.TemporaryDirectory(prefix="focused-", dir=frames.parent) as folder:
            cache = Path(folder)
            with (frames.parent / "tracking_status.csv").open("w", newline="", encoding="utf-8") as stream:
                status = csv.writer(stream)
                status.writerow(["frame_index", "object_id", "state", "crop_x", "crop_y", "crop_width", "crop_height", "missing_streak", "raw_area", "accepted_area"])
                for prompt in prompts:
                    self._track_object(files, prompt, width, height, cache, status)
            for index in range(len(files)):
                yield index, {p.object_id: read_image(cache / f"object_{p.object_id}" /
                             f"{index:06d}.png", 0) > 0 for p in prompts}

    def _track_object(self, files, prompt, width, height, cache, status):
        # Recentered segments overlap and used to decode their original JPEGs
        # repeatedly. Retain at most 16 frames / 128 MiB of decoded pixels per
        # object, never an unbounded full-video cache. Explicitly release the
        # pixels even when a caller retains an exception and its traceback.
        cache_frames = min(16, (128 * 1024 * 1024) // (width * height * 3))
        read_frame = lru_cache(maxsize=cache_frames)(read_image)
        try:
            self._track_object_cached(files, prompt, width, height, cache, status, read_frame)
        finally:
            read_frame.cache_clear()

    def _track_object_cached(self, files, prompt, width, height, cache, status, read_frame):
        object_id = prompt.object_id
        # Grow context for large selections; never silently clip a selected box.
        extent = max(prompt.box[2] - prompt.box[0], prompt.box[3] - prompt.box[1]) if prompt.box else 0
        if prompt.points:
            positive = np.array([p for p, label in zip(prompt.points, prompt.labels) if label])
            if len(positive) > 1:
                extent = max(extent, float(np.ptp(positive, axis=0).max()))
        side = max(self.crop_size, int(np.ceil(extent * 3)))
        center = prompt_center(prompt)
        current_prompt = prompt
        previous = None
        reference_area = None
        last_good_index = None
        missing_streak = 0
        lost = False
        start = 0
        exported = 0
        while exported < len(files):
            bounds = crop_bounds(center, side, width, height)
            x, y, right, bottom = bounds
            # Restart from the last GOOD frame, not an unrecognized overlap.
            # Extend past exported frames so recovery cannot loop without progress.
            end = min(len(files), max(start + self.segment_frames + 1,
                                      exported + self.segment_frames))
            with tempfile.TemporaryDirectory(prefix=f"crop-{object_id}-", dir=cache) as folder:
                crops = Path(folder)
                for index in range(start, end):
                    frame = read_frame(files[index])
                    write_image(crops / f"{index-start:06d}.jpg", frame[y:bottom, x:right])
                predictions = iter(self.tracker.predict(crops, [local_prompt(current_prompt, bounds)]))
                seen = 0
                try:
                    for local_index, masks in predictions:
                        if local_index != seen or start + seen >= end or set(masks) != {object_id}:
                            raise RuntimeError("Crop tracker returned invalid frames or IDs")
                        seen += 1
                        index = start + local_index
                        raw = np.asarray(masks[object_id], dtype=bool)
                        if raw.shape != (bottom-y, right-x):
                            raise RuntimeError("Crop mask has invalid dimensions")
                        if index < exported:  # overlap frame initializes the new SAM memory
                            continue
                        if index != exported:
                            raise RuntimeError("Crop tracking skipped a frame")
                        anchor = (center[0]-x, center[1]-y)
                        support = previous
                        if previous is not None:
                            aligned = motion_support(read_frame(files[last_good_index]),
                                                     read_frame(files[index]), previous, side)
                            if aligned is not None:
                                support = aligned
                                sy, sx = np.nonzero(aligned)
                                anchor = (float(sx.mean())-x, float(sy.mean())-y)
                        local_support = support[y:bottom, x:right] if support is not None else None
                        if local_support is not None and not local_support.any():
                            component = np.zeros_like(raw)
                        else:
                            component = target_regions(raw, anchor, local_support,
                                int(previous.sum()) if previous is not None else None,
                                max_step=side / 3 if previous is not None else None,
                                reference_area=reference_area)
                        full = np.zeros((height, width), bool)
                        full[y:bottom, x:right] = component
                        if not full.any():
                            missing_streak += 1
                            if object_id not in self.missing_object_ids:
                                self.missing_object_ids.append(object_id)
                            lost = previous is None or missing_streak >= self.recovery_frames
                            state = "lost" if lost else "missing"
                            if lost:
                                self.lost_object_ids.append(object_id)
                        else:
                            state = "recovered" if missing_streak else "tracked"
                            missing_streak = 0
                        write_image(cache / f"object_{object_id}" / f"{index:06d}.png", full.astype(np.uint8)*255)
                        status.writerow([index, object_id, state, x, y, right-x, bottom-y,
                                         missing_streak, int(raw.sum()), int(full.sum())])
                        exported += 1
                        print(f"Focused object {object_id}: {exported}/{len(files)}", flush=True)
                        if lost:
                            break
                        if not full.any():
                            continue
                        previous = full
                        last_good_index = index
                        reference_area = max(reference_area or 0, int(full.sum()))
                        ys, xs = np.nonzero(full)
                        center = (float(xs.mean()), float(ys.mean()))
                        margin = min(right-x, bottom-y) * .15
                        # Image borders cannot be recentered beyond the image.
                        near_edge = ((xs.min()-x < margin and x > 0) or
                                     (right-1-xs.max() < margin and right < width) or
                                     (ys.min()-y < margin and y > 0) or
                                     (bottom-1-ys.max() < margin and bottom < height))
                        if near_edge and exported < len(files):
                            break
                    else:
                        if seen != end-start:
                            raise RuntimeError("Crop tracker stopped early")
                finally:
                    if hasattr(predictions, "close"):
                        predictions.close()
            if lost:
                for index in range(exported, len(files)):
                    write_image(cache / f"object_{object_id}" / f"{index:06d}.png", np.zeros((height,width),np.uint8))
                    status.writerow([index, object_id, "lost", x, y, right-x, bottom-y,
                                     missing_streak, "", 0])
                return
            if exported < len(files):
                start = last_good_index
                current_prompt = mask_prompt(object_id, previous)
