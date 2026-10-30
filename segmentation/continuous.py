"""Bounded full-scene SAM memory, multi-object association and visible-only masks."""

import csv
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .focused import prompt_center, target_regions, target_component
from .motion import motion_support, appearance_histogram, appearance_matches
from .video import read_image


class ContinuousTracker:
    tracking_mode = "continuous-full-frame"

    def __init__(self, tracker, window_frames=48, recovery_frames=60):
        if type(window_frames) is not int or not 1 <= window_frames <= 60:
            raise ValueError("window_frames must be 1..60")
        if type(recovery_frames) is not int or not 1 <= recovery_frames <= 60:
            raise ValueError("recovery_frames must be 1..60")
        self.tracker = tracker
        self.window_frames, self.recovery_frames = window_frames, recovery_frames
        self.appearance_distance_limit = .7
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
        ids = {p.object_id for p in prompts}
        if not ids or len(ids) != len(prompts):
            raise ValueError("Unique selected object IDs are required")
        self.lost_object_ids = []
        self.missing_object_ids = []
        previous = {}
        good_frame = {}
        peak_area = {}
        reference_appearance = {}
        streak = {i: 0 for i in ids}
        cache_frames = min(16, (128*1024*1024)//(width*height*3))
        read_frame = lru_cache(maxsize=cache_frames)(read_image)
        exported = 0
        try:
            with (frames.parent / "tracking_status.csv").open("w", newline="", encoding="utf-8") as stream:
                status = csv.writer(stream)
                status.writerow(["frame_index", "object_id", "state", "crop_x", "crop_y", "crop_width",
                                 "crop_height", "missing_streak", "raw_area", "accepted_area"])
                while exported < len(files):
                    active = [p for p in prompts if p.object_id not in self.lost_object_ids]
                    if not active:
                        for index in range(exported, len(files)):
                            masks = {oid: np.zeros((height, width), bool) for oid in ids}
                            for oid in sorted(ids):
                                status.writerow([index, oid, "lost", 0, 0, width, height, streak[oid], "", 0])
                            yield index, masks
                        return
                    start = min(good_frame[p.object_id] for p in active) if exported else 0
                    end = min(len(files), exported+self.window_frames)
                    # Every seed is at most recovery_frames behind the next output.
                    if end-start > self.window_frames+self.recovery_frames:
                        raise RuntimeError("Continuation window exceeded its frame budget")
                    with tempfile.TemporaryDirectory(prefix="continuous-", dir=frames.parent) as folder:
                        window = Path(folder)
                        if not window.resolve().is_relative_to(frames.parent.resolve()):
                            raise RuntimeError("Temporary window is outside the output directory")
                        for index in range(start, end):
                            shutil.copyfile(files[index], window/f"{index-start:06d}.jpg")
                        kwargs = {}
                        if exported:
                            kwargs = {"initial_masks": {p.object_id: previous[p.object_id] for p in active},
                                      "prompt_frames": {p.object_id: good_frame[p.object_id]-start for p in active}}
                        predictions = iter(self.tracker.predict(window, active, **kwargs))
                        seen = 0
                        try:
                            for local_index, raw_masks in predictions:
                                if local_index != seen or start+seen >= end or set(raw_masks) != {p.object_id for p in active}:
                                    raise RuntimeError("Continuous tracker returned invalid frames or IDs")
                                seen += 1
                                index = start+local_index
                                if index < exported:
                                    continue
                                if index != exported:
                                    raise RuntimeError("Continuous tracking skipped a frame")
                                masks = {}
                                for prompt in prompts:
                                    oid = prompt.object_id
                                    if oid in self.lost_object_ids:
                                        masks[oid] = np.zeros((height, width), bool)
                                        status.writerow([index, oid, "lost", 0, 0, width, height, streak[oid], "", 0])
                                        continue
                                    raw = np.asarray(raw_masks[oid], dtype=bool)
                                    if raw.shape != (height, width):
                                        raise RuntimeError("Continuous mask dimensions do not match the video")
                                    old = previous.get(oid)
                                    anchor = prompt_center(prompt)
                                    support = old
                                    max_step = None
                                    if old is not None:
                                        ys, xs = np.nonzero(old)
                                        context = max(256, int(max(np.ptp(xs)+1, np.ptp(ys)+1)*3))
                                        aligned = motion_support(read_frame(files[good_frame[oid]]),
                                                                 read_frame(files[index]), old, context)
                                        support = aligned if aligned is not None else old
                                        sy, sx = np.nonzero(support)
                                        anchor = (float(sx.mean()), float(sy.mean()))
                                        max_step = context/3
                                    accepted = target_regions(raw, anchor, support,
                                                              int(old.sum()) if old is not None else None,
                                                              max_step, peak_area.get(oid))
                                    if old is not None and streak[oid] and raw.any() and not accepted.any():
                                        # Global SAM memory keeps the ID across an occlusion.
                                        # Permit a nearby return only with matching appearance;
                                        # never export the flow estimate as segmentation.
                                        candidate = target_component(raw, anchor, int(old.sum()),
                                                                     max_step, peak_area.get(oid))
                                        if candidate.any() and appearance_matches(
                                                reference_appearance.get(oid), read_frame(files[index]),
                                                candidate, self.appearance_distance_limit):
                                            accepted = candidate
                                    if accepted.any() and reference_appearance.get(oid) is not None:
                                        if not appearance_matches(reference_appearance[oid], read_frame(files[index]),
                                                                  accepted, self.appearance_distance_limit):
                                            accepted = np.zeros_like(accepted)
                                    if accepted.any():
                                        state = "recovered" if streak[oid] else "tracked"
                                        streak[oid] = 0
                                        previous[oid] = accepted
                                        good_frame[oid] = index
                                        peak_area[oid] = max(peak_area.get(oid, 0), int(accepted.sum()))
                                        if oid not in reference_appearance:
                                            reference_appearance[oid] = appearance_histogram(read_frame(files[index]), accepted)
                                    else:
                                        streak[oid] += 1
                                        if oid not in self.missing_object_ids:
                                            self.missing_object_ids.append(oid)
                                        lost = old is None or streak[oid] >= self.recovery_frames
                                        state = "lost" if lost else "missing"
                                        if lost:
                                            self.lost_object_ids.append(oid)
                                    masks[oid] = accepted
                                    status.writerow([index, oid, state, 0, 0, width, height,
                                                     streak[oid], int(raw.sum()), int(accepted.sum())])
                                exported += 1
                                print(f"Continuous objects: {exported}/{len(files)}", flush=True)
                                yield index, masks
                            if seen != end-start:
                                raise RuntimeError("Continuous tracker stopped early")
                        finally:
                            if hasattr(predictions, "close"):
                                predictions.close()
        finally:
            read_frame.cache_clear()
