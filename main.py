"""Command-line entry point; the same pipeline supports each project stage."""

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return value


def parser():
    root = argparse.ArgumentParser(description="User-guided SAM 2.1 video segmentation")
    commands = root.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Select objects on the first frame")
    select.add_argument("--video", type=Path, required=True)
    select.add_argument("--save", type=Path, required=True)
    select.add_argument("--single", action="store_true")
    run = commands.add_parser("run", help="Segment and export a video")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--prompts", type=Path, help="Saved selection; omit to select with a mouse")
    run.add_argument("--single", action="store_true", help="Limit the pilot to one object")
    run.add_argument("--model", choices=["small", "base-plus", "large"], default="small")
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    limits = run.add_mutually_exclusive_group()
    limits.add_argument("--max-frames", type=positive_integer,
                        help="Override the default 60-second limit with a frame count")
    limits.add_argument("--max-seconds", type=positive_integer,
                        help="Process at most this many seconds (default 60)")
    limits.add_argument("--all-frames", action="store_true", help="Process the entire video")
    run.add_argument("--fill-holes", type=nonnegative_integer, default=0, metavar="MAX_AREA")
    run.add_argument("--min-area", type=nonnegative_integer, default=0,
                     help="Remove components smaller than this area; default 0 retains tiny objects")
    run.add_argument("--tracks", action="store_true")
    run.add_argument("--transparent", action="store_true")
    run.add_argument("--small-objects", action="store_true",
                     help="Track selected objects with motion checks, bounded SAM windows and recovery")
    run.add_argument("--tracking-backend", choices=["continuous", "local"], default="continuous",
                     help="Small-object backend: continuous full-scene memory (default) or legacy local crops")
    run.add_argument("--crop-size", type=positive_integer, default=256,
                     help="For --tracking-backend local: crop side in source pixels (64..2048, default 256)")
    run.add_argument("--recovery-frames", type=positive_integer, default=60,
                     help="Consecutive missing frames before ending a small-object track (1..60, default 60)")
    evaluate = commands.add_parser("evaluate", help="Compute IoU and Dice on manually labeled frames")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument("--csv", type=Path, required=True)
    return root


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        # VS Code's Run Python File button provides no command-line arguments.
        # Resolve default files relative to this script, not the terminal folder.
        project = Path(__file__).resolve().parent
        video = project / "input" / "sample.mp4"
        output = project / "outputs" / f"selected-{uuid4().hex}"
        arguments = ["run", "--video", str(video), "--output", str(output),
                     "--small-objects", "--tracks", "--transparent", "--max-seconds", "60"]
        print("기본 실행: 최대 3개 선택 (숫자 1/2/3: 객체 선택, Enter: 시작). 영상의 앞 1분을 처리합니다.", flush=True)
        print(f"입력 영상: {video}\n결과 폴더: {output}", flush=True)
    args = parser().parse_args(arguments)
    try:
        if args.command == "evaluate":
            from segmentation.evaluation import evaluate
            print(json.dumps(evaluate(args.predictions, args.truth, args.csv), indent=2))
            return
        from segmentation.video import first_frame
        from segmentation.prompts import load_prompts, save_prompts
        frame = first_frame(args.video)
        height, width = frame.shape[:2]
        if args.command == "select":
            if args.save.exists():
                raise FileExistsError(f"Selection file already exists: {args.save}")
            from segmentation.selector import select_objects
            prompts = select_objects(frame, args.single)
            args.save.parent.mkdir(parents=True, exist_ok=True)
            save_prompts(args.save, prompts, width, height)
            print(f"Saved selection: {args.save}")
            return
        if args.output.exists():
            raise FileExistsError(f"Output already exists; choose a new directory: {args.output}")
        from segmentation.tracker import Sam2Tracker, MODELS
        checkpoint = args.checkpoint or Path(__file__).parent / "checkpoints" / MODELS[args.model][1]
        if args.prompts:
            prompts = load_prompts(args.prompts, width, height)
        else:
            from segmentation.selector import select_objects
            prompts = select_objects(frame, args.single)
        if args.single and len(prompts) != 1:
            raise ValueError("--single requires exactly one selected object")
        if args.small_objects and not 64 <= args.crop_size <= 2048:
            raise ValueError("--crop-size must be 64..2048")
        if args.small_objects and not 1 <= args.recovery_frames <= 60:
            raise ValueError("--recovery-frames must be 1..60")
        print(f"Selected {len(prompts)} object(s): {[p.object_id for p in prompts]}", flush=True)
        tracker = Sam2Tracker(checkpoint, args.model, args.device)
        if args.small_objects:
            if args.tracking_backend == "local":
                from segmentation.focused import FocusedTracker
                tracker = FocusedTracker(tracker, crop_size=args.crop_size,
                                         recovery_frames=args.recovery_frames)
            else:
                from segmentation.continuous import ContinuousTracker
                print("Tracking backend: continuous full-frame SAM memory", flush=True)
                tracker = ContinuousTracker(tracker, recovery_frames=args.recovery_frames)
        from segmentation.pipeline import process_video
        result = process_video(args.video, args.output, prompts, tracker,
                               max_frames=None if args.all_frames else args.max_frames,
                               max_seconds=None if args.all_frames or args.max_frames is not None else
                                           (args.max_seconds if args.max_seconds is not None else 60),
                               max_hole_area=args.fill_holes, min_component_area=args.min_area,
                               tracks=args.tracks, transparent=args.transparent)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise SystemExit(f"실행 실패: {error}") from error


if __name__ == "__main__":
    main()
