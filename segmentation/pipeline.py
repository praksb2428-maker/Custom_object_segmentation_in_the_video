"""End-to-end video processing with raw masks retained for fair comparisons."""

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .masks import cleanup_mask, centroid, render_frame
from .prompts import save_prompts, validate_prompts
from .video import extract_frames, make_writer, read_image, verify_video, write_image


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_video(video: Path, output: Path, prompts, tracker, *, max_frames=None, max_seconds=None,
                  max_hole_area=0, min_component_area=0, tracks=False, transparent=False):
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if max_hole_area < 0 or min_component_area < 0:
        raise ValueError("Cleanup thresholds cannot be negative")
    start_time = time.perf_counter()
    output.mkdir(parents=True)
    frame_dir = output / "frames"
    info = extract_frames(video, frame_dir, max_frames, max_seconds=max_seconds)
    validate_prompts(prompts, info.width, info.height)
    save_prompts(output / "prompts.json", prompts, info.width, info.height)
    ids = {prompt.object_id for prompt in prompts}
    trajectories = {}
    writers = []
    count = 0
    try:
        foreground_writer = make_writer(output / "foreground.mp4", info)
        writers.append(foreground_writer)
        overlay_writer = make_writer(output / "overlay.mp4", info)
        writers.append(overlay_writer)
        with (output / "centroids.csv").open("w", encoding="utf-8", newline="") as stream:
            table = csv.writer(stream)
            table.writerow(["frame_index", "time_seconds", "object_id", "area_pixels", "cx", "cy"])
            predictions = iter(tracker.predict(frame_dir, prompts))
            try:
                for frame_index, raw_masks in predictions:
                    if frame_index != count or count >= info.frame_count:
                        raise RuntimeError("Tracker returned missing, repeated or out-of-order frames")
                    if set(raw_masks) != ids:
                        raise RuntimeError("Tracker returned a different set of object IDs")
                    frame = read_image(frame_dir / f"{frame_index:06d}.jpg")
                    masks = {}
                    for object_id, raw_mask in raw_masks.items():
                        raw = np.asarray(raw_mask, dtype=bool)
                        if raw.shape != (info.height, info.width):
                            raise RuntimeError("Tracker mask dimensions do not match the video")
                        corrected = cleanup_mask(raw, max_hole_area, min_component_area)
                        name = f"{frame_index:06d}.png"
                        write_image(output / "masks_raw" / f"object_{object_id}" / name, raw.astype(np.uint8) * 255)
                        write_image(output / "masks" / f"object_{object_id}" / name, corrected.astype(np.uint8) * 255)
                        if transparent:
                            rgba = np.dstack((frame, corrected.astype(np.uint8) * 255))
                            rgba[~corrected, :3] = 0
                            write_image(output / "transparent" / f"object_{object_id}" / name, rgba)
                        masks[object_id] = corrected
                        center = centroid(corrected)
                        table.writerow([frame_index, frame_index / info.fps, object_id,
                                        int(corrected.sum()), *(center if center else ("", ""))])
                    foreground, overlay = render_frame(frame, masks, trajectories, tracks)
                    foreground_writer.write(foreground)
                    overlay_writer.write(overlay)
                    count += 1
                    print(f"Processed {count}/{info.frame_count}", flush=True)
            finally:
                if hasattr(predictions, "close"):
                    predictions.close()
        if count != info.frame_count:
            raise RuntimeError(f"Tracker stopped early: {count}/{info.frame_count}")
    finally:
        for writer in writers:
            writer.release()
    verify_video(output / "foreground.mp4", info)
    verify_video(output / "overlay.mp4", info)
    elapsed = time.perf_counter() - start_time
    run = {"schema_version": 1, "status": "completed", "input_sha256": file_hash(video),
           "video": asdict(info), "object_ids": sorted(ids), "elapsed_seconds": elapsed,
           "end_to_end_fps": count / elapsed, "model": getattr(tracker, "model", "test-double"),
           "device": getattr(tracker, "device", "test-double"),
           "max_hole_area": max_hole_area, "min_component_area": min_component_area,
           "tracks": tracks, "transparent": transparent,
           "audio_preserved": False, "timing_scope": "extraction_through_export_verification; excludes_model_load"}
    run["tracking_mode"] = getattr(tracker, "tracking_mode", "full-frame")
    if hasattr(tracker, "recovery_frames"):
        if hasattr(tracker, "crop_size"):
            run["crop_size"] = tracker.crop_size
            run["segment_frames"] = tracker.segment_frames
        if hasattr(tracker, "window_frames"):
            run["window_frames"] = tracker.window_frames
            run["max_window_frames"] = tracker.window_frames+tracker.recovery_frames
            run["appearance_distance_limit"] = tracker.appearance_distance_limit
            run["appearance_reference"] = "first_accepted_frame; requires_at_least_four_pixels"
        run["recovery_frames"] = tracker.recovery_frames
        run["lost_object_ids"] = tracker.lost_object_ids
        run["missing_object_ids"] = tracker.missing_object_ids
        run["tracking_status_schema_version"] = 2
        if tracker.lost_object_ids:
            print(f"WARNING: Lost target IDs {tracker.lost_object_ids}; see tracking_status.csv", flush=True)
        elif tracker.missing_object_ids:
            print(f"WARNING: Targets had missing frames {tracker.missing_object_ids}; see tracking_status.csv", flush=True)
    if getattr(tracker, "checkpoint", None):
        run["checkpoint_sha256"] = file_hash(tracker.checkpoint)
    (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return run
