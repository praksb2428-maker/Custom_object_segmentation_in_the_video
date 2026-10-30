"""Serializable first-frame prompts and zoom-to-original coordinate mapping."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

MAX_OBJECTS = 3


@dataclass
class ObjectPrompt:
    object_id: int
    box: list[float] | None = None
    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def validate(self, width: int, height: int) -> None:
        if type(self.object_id) is not int or self.object_id <= 0:
            raise ValueError("object_id must be a positive integer")
        if len(self.points) != len(self.labels):
            raise ValueError("points and labels must have equal lengths")
        if self.box is None and not self.points:
            raise ValueError("Select a box or at least one foreground point")
        if self.box is not None:
            if len(self.box) != 4 or not all(math.isfinite(v) for v in self.box):
                raise ValueError("box must contain four finite coordinates")
            x1, y1, x2, y2 = self.box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("box is outside the original image or has zero area")
        if self.box is None and 1 not in self.labels:
            raise ValueError("Point-only prompts need a foreground point")
        for point, label in zip(self.points, self.labels):
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError("Each point must contain two finite coordinates")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("point is outside the original image")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("Point label must be 1 (foreground) or 0 (background)")


def validate_prompts(prompts: list[ObjectPrompt], width: int, height: int) -> None:
    if not prompts:
        raise ValueError("At least one object must be selected")
    if len(prompts) > MAX_OBJECTS:
        raise ValueError(f"Select at most {MAX_OBJECTS} objects")
    ids = [prompt.object_id for prompt in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("object_id values must be unique")
    for prompt in prompts:
        prompt.validate(width, height)


def load_prompts(path: Path, width: int, height: int) -> list[ObjectPrompt]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported prompt schema_version")
    if document.get("frame_index", 0) != 0:
        raise ValueError("This version accepts prompts on the first frame only")
    if document.get("width") != width or document.get("height") != height:
        raise ValueError("Prompt dimensions do not match the input video")
    prompts = [ObjectPrompt(**entry) for entry in document["objects"]]
    validate_prompts(prompts, width, height)
    return prompts


def save_prompts(path: Path, prompts: list[ObjectPrompt], width: int, height: int) -> None:
    from dataclasses import asdict
    validate_prompts(prompts, width, height)
    document = {"schema_version": 1, "frame_index": 0, "width": width,
                "height": height, "objects": [asdict(prompt) for prompt in prompts]}
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ViewTransform:
    """A rectangular original-image crop displayed at a fixed canvas size."""

    x: float
    y: float
    width: float
    height: float
    canvas_width: int
    canvas_height: int

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        return (self.x + x * self.width / self.canvas_width,
                self.y + y * self.height / self.canvas_height)

    def to_canvas(self, x: float, y: float) -> tuple[int, int]:
        return (round((x - self.x) * self.canvas_width / self.width),
                round((y - self.y) * self.canvas_height / self.height))

    def zoom(self, x: int, y: int, factor: float,
             image_width: int, image_height: int) -> "ViewTransform":
        anchor_x, anchor_y = self.to_original(x, y)
        # Match the exact integer crop used by the selection window.
        new_width = min(image_width, max(math.ceil(image_width / 16), round(self.width / factor)))
        new_height = min(image_height, max(math.ceil(image_height / 16), round(self.height / factor)))
        left = anchor_x - (x / self.canvas_width) * new_width
        top = anchor_y - (y / self.canvas_height) * new_height
        left = round(max(0, min(image_width - new_width, left)))
        top = round(max(0, min(image_height - new_height, top)))
        return ViewTransform(left, top, new_width, new_height,
                             self.canvas_width, self.canvas_height)
