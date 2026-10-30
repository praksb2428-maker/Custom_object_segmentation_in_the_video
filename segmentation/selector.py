"""Native first-frame selection UI; zoom changes only the display coordinates."""

import cv2
from .prompts import MAX_OBJECTS, ObjectPrompt, ViewTransform, validate_prompts


def select_objects(frame, single: bool = False) -> list[ObjectPrompt]:
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 720 / height)
    canvas_width, canvas_height = max(1, round(width * scale)), max(1, round(height * scale))
    full_view = ViewTransform(0, 0, width, height, canvas_width, canvas_height)
    view = full_view
    drag_start = None
    limit = 1 if single else MAX_OBJECTS
    objects = [ObjectPrompt(object_id) for object_id in range(1, limit + 1)]
    current = objects[0]
    colors = [(80, 220, 80), (0, 200, 255), (255, 180, 100)]
    window = f"Select up to {limit} objects | 1/2/3: choose object | Enter: finish | U: undo | R: zoom reset | Esc: cancel"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, _):
        nonlocal view, drag_start
        if not (0 <= x < canvas_width and 0 <= y < canvas_height):
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.5 if flags > 0 else 1 / 1.5
            view = view.zoom(x, y, factor, width, height)
        elif event == cv2.EVENT_LBUTTONDOWN:
            drag_start = view.to_original(x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            end = view.to_original(x, y)
            start = drag_start
            drag_start = None
            if abs(end[0] - start[0]) * canvas_width / view.width < 4 and abs(end[1] - start[1]) * canvas_height / view.height < 4:
                current.points.append(list(end))
                current.labels.append(1)
            else:
                current.box = [min(start[0], end[0]), min(start[1], end[1]),
                               max(start[0], end[0]), max(start[1], end[1])]
        elif event == cv2.EVENT_RBUTTONUP:
            current.points.append(list(view.to_original(x, y)))
            current.labels.append(0)

    cv2.setMouseCallback(window, mouse)
    print(f"객체를 최대 {limit}개 선택할 수 있습니다.", flush=True)
    if single:
        print("물체 하나를 선택하고 Enter를 누르세요.", flush=True)
    else:
        print("숫자 1 → 물체 1 선택, 숫자 2 → 물체 2 선택, 숫자 3 → 물체 3 선택, Enter → 시작", flush=True)
        print("1/2/3으로 이전 객체를 다시 수정할 수 있습니다. N 또는 스페이스는 다음 객체입니다.", flush=True)
        print("1~2개만 선택해도 Enter로 시작할 수 있습니다.", flush=True)
    print("여러 번 클릭하면 같은 객체의 포함 지점이 추가됩니다.", flush=True)
    print("Drag: box | Left: foreground | Right: background | Wheel: zoom | U: undo | R: reset | Esc: cancel")
    try:
        while True:
            crop = frame[int(view.y):max(int(view.y) + 1, int(view.y + view.height)),
                         int(view.x):max(int(view.x) + 1, int(view.x + view.width))]
            canvas = cv2.resize(crop, (canvas_width, canvas_height))
            for prompt in objects:
                color = colors[(prompt.object_id - 1) % len(colors)]
                if prompt.box:
                    a = view.to_canvas(*prompt.box[:2])
                    b = view.to_canvas(*prompt.box[2:])
                    cv2.rectangle(canvas, a, b, color, 2)
                    cv2.putText(canvas, str(prompt.object_id), a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                for point, label in zip(prompt.points, prompt.labels):
                    cv2.circle(canvas, view.to_canvas(*point), 4,
                               color if label else (0, 0, 255), -1)
                if prompt.box is None and prompt.points:
                    cv2.putText(canvas, str(prompt.object_id), view.to_canvas(*prompt.points[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            selected = sum(prompt.box is not None or bool(prompt.points) for prompt in objects)
            keys = "1=choose" if single else "1/2/3=choose"
            hints = [(f"Editing object {current.object_id}/{limit} | {keys} | Enter=start", (8, 22), (255, 255, 255)),
                     (f"Selected: {selected}/{limit} | Left=include Right=exclude | Wheel=zoom | U=undo",
                      (8, 46), (0, 255, 255))]
            for text, position, color in hints:
                cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                raise ValueError("Object selection cancelled")
            if key in (ord("1"), ord("2"), ord("3")):
                object_id = key - ord("0")
                if object_id > limit:
                    print(f"현재 모드는 객체 {limit}개까지만 선택할 수 있습니다.", flush=True)
                    continue
                # Do not let a drag begun on one object finish on another.
                drag_start = None
                current = objects[object_id - 1]
                print(f"객체 {object_id} 선택 중 (기존 선택 유지)", flush=True)
            elif key == ord("r"):
                view = full_view
            elif key == ord("u"):
                if current.points:
                    current.points.pop()
                    current.labels.pop()
                else:
                    current.box = None
            elif key in (ord("n"), ord("N"), 32, 13, 10):
                if key in (13, 10):
                    selected_prompts = [prompt for prompt in objects
                                        if prompt.box is not None or bool(prompt.points)]
                    try:
                        validate_prompts(selected_prompts, width, height)
                    except ValueError as error:
                        print(error, flush=True)
                        continue
                    return selected_prompts
                has_input = current.box is not None or bool(current.points)
                if has_input:
                    try:
                        current.validate(width, height)
                    except ValueError as error:
                        print(error)
                        continue
                    if single:
                        return [current]
                    if current.object_id == limit:
                        print(f"최대 {limit}개까지 선택할 수 있습니다. Enter를 눌러 시작하세요.", flush=True)
                        continue
                    drag_start = None
                    current = objects[current.object_id]
                    print(f"객체 {current.object_id} 선택 중 (기존 선택 유지)", flush=True)
    finally:
        cv2.destroyWindow(window)
