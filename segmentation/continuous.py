"""Bounded full-scene SAM memory, multi-object association and visible-only masks."""

import csv
import hashlib
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .focused import prompt_center, target_regions, target_component
from .motion import motion_support, appearance_histogram, appearance_matches, AppearanceMemory
from .video import read_image
from .edge import supported_edge_shrink


class ContinuousTracker:
    tracking_mode = "continuous-full-frame"

    def __init__(self, tracker, window_frames=48, recovery_frames=60, adaptive_identity=True, enable_reacquisition=False, enable_small_refinement=False, enable_prompt_refinement=False, edge_partials=False):
        if type(window_frames) is not int or not 1 <= window_frames <= 60:
            raise ValueError("window_frames must be 1..60")
        if type(recovery_frames) is not int or not 1 <= recovery_frames <= 60:
            raise ValueError("recovery_frames must be 1..60")
        self.tracker = tracker
        self.window_frames, self.recovery_frames = window_frames, recovery_frames
        self.appearance_distance_limit = .7
        self.adaptive_identity = bool(adaptive_identity)
        self.enable_reacquisition = bool(enable_reacquisition)
        self.enable_small_refinement = bool(enable_small_refinement)
        self.small_refiner = None
        self.edge_partials = bool(edge_partials)
        self.edge_events = []
        self.edge_event_count = 0
        self.enable_prompt_refinement = bool(enable_prompt_refinement)
        self.prompt_refiner = None
        for name in ("model", "device", "checkpoint"):
            if hasattr(tracker, name):
                setattr(self, name, getattr(tracker, name))
        self.lost_object_ids = []
        self.missing_object_ids = []
        self.reacquired_object_ids = []
        self.ever_lost_object_ids = []

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
        self.reacquired_object_ids = []
        self.ever_lost_object_ids = []
        if self.enable_small_refinement:
            from .refinement import SmallObjectRefiner
            self.small_refiner = SmallObjectRefiner(self.tracker)
        else:
            self.small_refiner = None
        if self.enable_prompt_refinement:
            from .refinement import ReferenceMaskRefiner
            self.prompt_refiner = ReferenceMaskRefiner(self.tracker)
        else:
            self.prompt_refiner = None
        self.edge_events = []
        self.edge_event_count = 0
        previous = {}
        good_frame = {}
        peak_area = {}
        reference_appearance = {}
        appearance_models = {}
        initial_area = {}
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
                    live = [p for p in prompts if p.object_id not in self.lost_object_ids]
                    active = [p for p in prompts if p in live or
                              (self.enable_reacquisition and p.object_id in previous)]
                    if not active:
                        for index in range(exported, len(files)):
                            masks = {oid: np.zeros((height, width), bool) for oid in ids}
                            for oid in sorted(ids):
                                status.writerow([index, oid, "lost", 0, 0, width, height, streak[oid], "", 0])
                            yield index, masks
                        return
                    end = min(len(files), exported+self.window_frames)
                    start = min((good_frame[p.object_id] for p in live), default=exported) if exported else 0
                    indices = list(range(start, end))
                    if exported and self.enable_reacquisition:
                        # Dormant targets contribute only one actual reference frame, not
                        # their entire missing history. New output remains contiguous.
                        indices = sorted(set(indices) | {good_frame[p.object_id] for p in active})
                    budget = self.window_frames+self.recovery_frames+(len(prompts) if self.enable_reacquisition else 0)
                    if len(indices) > budget:
                        raise RuntimeError("Continuation window exceeded its frame budget")
                    positions = {index: local for local, index in enumerate(indices)}
                    with tempfile.TemporaryDirectory(prefix="continuous-", dir=frames.parent) as folder:
                        window = Path(folder)
                        if not window.resolve().is_relative_to(frames.parent.resolve()):
                            raise RuntimeError("Temporary window is outside the output directory")
                        for local, index in enumerate(indices):
                            shutil.copyfile(files[index], window/f"{local:06d}.jpg")
                        kwargs = {}
                        if exported:
                            seed_masks = {p.object_id: previous[p.object_id] for p in active}
                            if self.prompt_refiner is not None:
                                from .refinement import reference_prompt
                                from .focused import mask_prompt
                                # Snapshot actual same-frame rivals before any seed is updated.
                                old_seeds = dict(seed_masks)
                                for prompt in active:
                                    oid = prompt.object_id
                                    seed_index = good_frame[oid]
                                    rivals = {other_id: mask for other_id, mask in old_seeds.items()
                                              if good_frame[other_id] == seed_index}
                                    rival_prompts = [mask_prompt(other_id, mask) for other_id, mask in rivals.items()]
                                    seed = old_seeds[oid]
                                    corrected = self.prompt_refiner.refine(
                                        read_frame(files[seed_index]), reference_prompt(oid, seed),
                                        seed, rival_prompts, rivals)
                                    if corrected is not seed and appearance_models[oid].matches(
                                            read_frame(files[seed_index]), corrected, self.appearance_distance_limit):
                                        seed_masks[oid] = corrected
                                        previous[oid] = corrected
                                        peak_area[oid] = max(peak_area[oid], int(corrected.sum()))
                                        self.prompt_refiner.applied += 1
                            kwargs = {"initial_masks": seed_masks,
                                      "prompt_frames": {p.object_id: positions[good_frame[p.object_id]] for p in active}}
                        predictions = iter(self.tracker.predict(window, active, **kwargs))
                        seen = 0
                        try:
                            for local_index, raw_masks in predictions:
                                if local_index != seen or seen >= len(indices) or set(raw_masks) != {p.object_id for p in active}:
                                    raise RuntimeError("Continuous tracker returned invalid frames or IDs")
                                seen += 1
                                index = indices[local_index]
                                if index < exported:
                                    continue
                                if index != exported:
                                    raise RuntimeError("Continuous tracking skipped a frame")
                                masks = {}
                                for prompt in prompts:
                                    oid = prompt.object_id
                                    if oid in self.lost_object_ids and (not self.enable_reacquisition or oid not in previous):
                                        masks[oid] = np.zeros((height, width), bool)
                                        status.writerow([index, oid, "lost", 0, 0, width, height, streak[oid], "", 0])
                                        continue
                                    raw = np.asarray(raw_masks[oid], dtype=bool)
                                    if raw.shape != (height, width):
                                        raise RuntimeError("Continuous mask dimensions do not match the video")
                                    raw_area = int(raw.sum())
                                    old = previous.get(oid)
                                    anchor = prompt_center(prompt)
                                    support = old
                                    max_step = None
                                    aligned = None
                                    if old is not None:
                                        ys, xs = np.nonzero(old)
                                        context = max(256, int(max(np.ptp(xs)+1, np.ptp(ys)+1)*3))
                                        aligned = motion_support(read_frame(files[good_frame[oid]]),
                                                                 read_frame(files[index]), old, context)
                                        support = aligned if aligned is not None else old
                                        sy, sx = np.nonzero(support)
                                        anchor = (float(sx.mean()), float(sy.mean()))
                                        max_step = context/3
                                        if self.enable_reacquisition and streak[oid]:
                                            max_step = min(float(np.hypot(width, height)), max_step+
                                                           np.sqrt(streak[oid])*max(4, context/30))
                                    area_reference = peak_area.get(oid)
                                    if self.adaptive_identity and aligned is not None and good_frame[oid] == index-1:
                                        # A checked one-frame scale estimate is more relevant than an old peak.
                                        area_reference = int(aligned.sum())
                                    historical_reference = area_reference
                                    consecutive = old is not None and good_frame.get(oid) == index-1 and streak[oid] == 0
                                    edge_raw = self.edge_partials and supported_edge_shrink(raw, old, consecutive=consecutive)
                                    if edge_raw:
                                        area_reference = int(old.sum())
                                    accepted = target_regions(raw, anchor, support,
                                                              int(old.sum()) if old is not None else None,
                                                              max_step, area_reference)
                                    if old is not None and streak[oid] and raw.any() and not accepted.any():
                                        # Global SAM memory keeps the ID across an occlusion.
                                        # Permit a nearby return only with matching appearance;
                                        # never export the flow estimate as segmentation.
                                        candidate = target_component(raw, anchor, int(old.sum()),
                                                                     max_step, area_reference)
                                        if candidate.any():
                                            matches = (appearance_models[oid].matches(read_frame(files[index]), candidate,
                                                       self.appearance_distance_limit) if self.adaptive_identity else
                                                       appearance_matches(reference_appearance.get(oid), read_frame(files[index]),
                                                                          candidate, self.appearance_distance_limit))
                                            if matches:
                                                accepted = candidate
                                    if self.small_refiner is not None and accepted.any():
                                        accepted = self.small_refiner.refine(read_frame(files[index]), accepted)
                                    edge_model_sha256 = hashlib.sha256(accepted.tobytes(order="C")).hexdigest() if edge_raw else None
                                    edge_final = edge_raw and supported_edge_shrink(accepted, old, consecutive=consecutive)
                                    if edge_raw and not edge_final:
                                        # A refiner cannot carry the exception to an unsupported fragment.
                                        original_regions = target_regions(raw, anchor, support, int(old.sum()), max_step, historical_reference)
                                        if not original_regions.any():
                                            accepted = np.zeros_like(accepted)
                                    if accepted.any() and reference_appearance.get(oid) is not None:
                                        matches = (appearance_models[oid].matches(read_frame(files[index]), accepted,
                                                   self.appearance_distance_limit) if self.adaptive_identity else
                                                   appearance_matches(reference_appearance[oid], read_frame(files[index]),
                                                                      accepted, self.appearance_distance_limit))
                                        if not matches:
                                            accepted = np.zeros_like(accepted)
                                    if self.adaptive_identity and oid in initial_area and int(accepted.sum()) < max(1, initial_area[oid]*.1) and not edge_final:
                                        accepted = np.zeros_like(accepted)
                                    if self.enable_reacquisition and streak[oid] and accepted.any():
                                        # A recovering ID must not claim a confidently visible other ID.
                                        # Compare against both already accepted and plausible live proposals;
                                        # prompt ordering must not weaken this guard.
                                        for other in active:
                                            other_id = other.object_id
                                            if other_id == oid or streak[other_id] or other_id not in previous:
                                                continue
                                            rival = masks.get(other_id, np.asarray(raw_masks[other_id], dtype=bool))
                                            if rival.shape != (height, width):
                                                raise RuntimeError("Continuous mask dimensions do not match the video")
                                            model = appearance_models.get(other_id)
                                            if rival.any() and model is not None and model.matches(read_frame(files[index]), rival):
                                                shared = int((accepted & rival).sum())
                                                if shared / max(1, min(int(accepted.sum()), int(rival.sum()))) > .5:
                                                    accepted = np.zeros_like(accepted)
                                                    break
                                    if edge_raw:
                                        self.edge_event_count += 1
                                        if len(self.edge_events) < 512:
                                            self.edge_events.append({"frame":index,"object_id":oid,"raw_area":raw_area,"previous_area":int(old.sum()),"historical_reference":historical_reference,"effective_reference":area_reference,"postrefine_eligible":bool(edge_final),"postrefine_model_mask_sha256":edge_model_sha256,"export_area":int(accepted.sum())})
                                    if accepted.any():
                                        state = "recovered" if streak[oid] else "tracked"
                                        if oid in self.lost_object_ids:
                                            self.lost_object_ids.remove(oid)
                                            if oid not in self.reacquired_object_ids:
                                                self.reacquired_object_ids.append(oid)
                                        if self.adaptive_identity and oid in appearance_models and old is not None and not streak[oid]:
                                            union = int((accepted | support).sum())
                                            overlap = int((accepted & support).sum()) / max(union, 1)
                                            appearance_models[oid].observe(read_frame(files[index]), accepted, overlap=overlap,
                                                                           area_ratio=int(accepted.sum())/max(int(old.sum()), 1))
                                        streak[oid] = 0
                                        previous[oid] = accepted
                                        good_frame[oid] = index
                                        peak_area[oid] = max(peak_area.get(oid, 0), int(accepted.sum()))
                                        if oid not in reference_appearance:
                                            reference_appearance[oid] = appearance_histogram(read_frame(files[index]), accepted)
                                            appearance_models[oid] = AppearanceMemory(read_frame(files[index]), accepted)
                                            initial_area[oid] = int(accepted.sum())
                                    else:
                                        streak[oid] += 1
                                        if oid not in self.missing_object_ids:
                                            self.missing_object_ids.append(oid)
                                        lost = old is None or streak[oid] >= self.recovery_frames
                                        state = "lost" if lost else "missing"
                                        if lost:
                                            if oid not in self.lost_object_ids:
                                                self.lost_object_ids.append(oid)
                                            if oid not in self.ever_lost_object_ids:
                                                self.ever_lost_object_ids.append(oid)
                                    masks[oid] = accepted
                                    status.writerow([index, oid, state, 0, 0, width, height,
                                                     streak[oid], raw_area, int(accepted.sum())])
                                exported += 1
                                print(f"Continuous objects: {exported}/{len(files)}", flush=True)
                                yield index, masks
                            if seen != len(indices):
                                raise RuntimeError("Continuous tracker stopped early")
                        finally:
                            if hasattr(predictions, "close"):
                                predictions.close()
        finally:
            read_frame.cache_clear()
