"""Frame extraction and Unicode-safe image IO on Windows."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        success, frame = capture.read()
        if not success:
            raise ValueError("Video has no decodable frames")
        return frame
    finally:
        capture.release()


def extract_frames(video: Path, directory: Path, max_frames: int | None = None,
                   *, max_seconds: float | None = None) -> VideoInfo:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if max_seconds is not None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        if max_frames is not None:
            raise ValueError("Choose max_frames or max_seconds, not both")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS is missing or invalid")
        if max_seconds is not None:
            max_frames = math.floor(fps * max_seconds)
            if max_frames < 1:
                raise ValueError("Duration limit is shorter than one video frame")
        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(reported_count) or reported_count < 1 or not reported_count.is_integer():
            raise ValueError("Video frame count is missing or invalid; use an indexed MP4")
        expected_count = int(reported_count)
        if max_frames is not None:
            expected_count = min(expected_count, max_frames)
        directory.mkdir(parents=True, exist_ok=False)
        count = 0
        dimensions = None
        while max_frames is None or count < max_frames:
            success, frame = capture.read()
            if not success:
                break
            height, width = frame.shape[:2]
            if dimensions is None:
                dimensions = (width, height)
            elif dimensions != (width, height):
                raise ValueError("Frame dimensions changed within the video")
            write_image(directory / f"{count:06d}.jpg", frame)
            count += 1
        if count == 0 or dimensions is None:
            raise ValueError("Video has no decodable frames")
        if count != expected_count:
            raise RuntimeError(f"Input decoding incomplete or metadata inconsistent: {count}/{expected_count} frames")
        return VideoInfo(dimensions[0], dimensions[1], fps, count)
    finally:
        capture.release()


def make_writer(path: Path, info: VideoInfo) -> cv2.VideoWriter:
    if info.width % 2 or info.height % 2:
        raise ValueError("MP4 export needs even frame dimensions; resize the video first")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             info.fps, (info.width, info.height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create MP4: {path}")
    return writer


def verify_video(path: Path, info: VideoInfo) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot reopen exported video: {path}")
        count = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[:2] != (info.height, info.width):
                raise RuntimeError("Exported video dimensions do not match the input")
            count += 1
        if count != info.frame_count:
            raise RuntimeError(f"Exported frame count mismatch: {count} != {info.frame_count}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if abs(fps - info.fps) > 0.02:
            raise RuntimeError("Exported FPS does not match the input")
    finally:
        capture.release()
