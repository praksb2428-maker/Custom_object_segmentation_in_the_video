"""Reproduce a selected-frame SAM 2 video experiment without application filters."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segmentation.prompts import ObjectPrompt, validate_prompts
from segmentation.tracker import MODELS, Sam2Tracker
from segmentation.video import extract_frames, make_writer, read_image, verify_video, write_image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path, object_ids=None):
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported selection schema_version')
    start = document['frame_index']
    if type(start) is not int or start < 0:
        raise ValueError('frame_index must be a nonnegative integer')
    prompts = [ObjectPrompt(**entry) for entry in document['objects']]
    validate_prompts(prompts, document['width'], document['height'])
    if object_ids is not None:
        if len(set(object_ids)) != len(object_ids) or not set(object_ids) <= {p.object_id for p in prompts}:
            raise ValueError('Object IDs must be unique and present in the selection')
        prompts = [p for p in prompts if p.object_id in object_ids]
        validate_prompts(prompts, document['width'], document['height'])
    return document, prompts


def run(args):
    document, prompts = load_selection(args.selection, args.object_ids)
    start = document["frame_index"]
    checkpoint = args.checkpoint or ROOT / 'checkpoints' / MODELS[args.model][1]
    if not args.video.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Prepare the input video and checkpoint before running')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten')
    input_sha = sha256(args.video)
    if document.get('video_sha256') and document['video_sha256'] != input_sha:
        raise ValueError('The selection belongs to a different video')
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True)
    frame_dir = args.output / '_frames'
    sam_dir = args.output / '_sam_frames'
    owned = []
    writers = {}
    iterator = None
    rows = []
    try:
        info = extract_frames(args.video, frame_dir)
        if (info.width, info.height) != (document["width"], document["height"]):
            raise ValueError("Selection dimensions do not match the video")
        if start >= info.frame_count:
            raise ValueError("Selection frame is outside the video")
        owned.extend(sorted(frame_dir.glob('*.jpg')))
        if (info.width, info.height) != (document['width'], document['height']):
            raise ValueError('Selection dimensions do not match the video')
        start = document['frame_index']
        if start >= info.frame_count:
            raise ValueError('Selection frame is outside the video')
        sam_dir.mkdir()
        for index in range(start, info.frame_count):
            target = sam_dir / f'{index-start:06d}.jpg'
            # Copy only model inputs; cleanup below never removes caller-owned files.
            shutil.copyfile(frame_dir / f'{index:06d}.jpg', target)
            owned.append(target)
        writers = {kind: make_writer(args.output / f'{kind}.mp4', info) for kind in ('overlay', 'foreground')}
        ids = {prompt.object_id for prompt in prompts}
        colors = [(0, 210, 255), (0, 220, 0), (255, 170, 40)]
        palette = {oid: colors[(oid-1) % 3] for oid in ids}
        empty = np.zeros((info.height, info.width), bool)

        def emit(frame_index, masks):
            image = read_image(frame_dir / f'{frame_index:06d}.jpg')
            overlay = image.copy()
            union = empty.copy()
            areas = {}
            for oid in sorted(ids):
                mask = masks.get(oid, empty)
                if mask.shape != empty.shape:
                    raise ValueError('SAM mask dimensions differ from the source')
                union |= mask
                areas[str(oid)] = int(mask.sum())
                write_image(args.output / f'masks/object_{oid}/{frame_index:06d}.png', mask.astype(np.uint8)*255)
                overlay[mask] = (overlay[mask]*.5 + np.asarray(palette[oid])*.5).astype(np.uint8)
                y, x = np.nonzero(mask)
                if len(x):
                    cv2.putText(overlay, f'ID {oid}', (int(x.mean()), int(y.mean())), cv2.FONT_HERSHEY_SIMPLEX, .35, palette[oid], 1)
            label = 'waiting' if frame_index < start else 'SAM2 video'
            cv2.putText(overlay, f'{frame_index/info.fps:.2f}s | {label}', (3, info.height-6), cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1)
            foreground = image.copy()
            foreground[~union] = 0
            writers['overlay'].write(overlay)
            writers['foreground'].write(foreground)
            rows.append({'frame': frame_index, 'areas': areas})

        for index in range(start):
            emit(index, {})
        tracker = Sam2Tracker(checkpoint, args.model, args.device)
        iterator = tracker.predict(sam_dir, prompts)
        next_index = start
        for relative_index, masks in iterator:
            index = relative_index + start
            if index != next_index or set(masks) != ids:
                raise RuntimeError('Unexpected frame order or object IDs')
            emit(index, masks)
            next_index += 1
        if next_index != info.frame_count:
            raise RuntimeError('SAM did not return the entire requested timeline')
    finally:
        if iterator is not None:
            iterator.close()
        for writer in writers.values():
            writer.release()
        # On partial extraction failure, these JPEGs are also owned by this run.
        if frame_dir.exists():
            owned.extend(frame_dir.glob('*.jpg'))
        if sam_dir.exists():
            owned.extend(sam_dir.glob('*.jpg'))
        for path in set(owned):
            if path.is_file():
                path.unlink()
        for directory in (sam_dir, frame_dir):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
    for kind in writers:
        verify_video(args.output / f'{kind}.mp4', info)
    result = {'schema_version': 1, 'status': 'completed', 'video_sha256': input_sha,
              'selection_sha256': sha256(args.selection), 'checkpoint_sha256': sha256(checkpoint),
              'runner_sha256': sha256(__file__), 'initial_frame': start, 'object_ids': sorted(ids),
              'video': asdict(info), 'mode': 'pure_sam2_video', 'fine_tuning': False,
              'waiting_prefix_frames': start, 'future_annotation_input': False,
              'source_video_sha256': input_sha, 'checkpoint_sha256': sha256(checkpoint),
              'selection_sha256': sha256(args.selection), 'selected_ids': sorted(ids),
              'initial_frame': start, 'video_info': asdict(info), 'fine_tuned': False,
              'runner_sha256': sha256(Path(__file__)), 'torch': torch.__version__,
              'rows': rows, 'note': 'Completion is output verification, not accuracy certification.'}
    (args.output / 'run.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'Completed: {args.output.resolve()}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object-ids', type=int, nargs='+')
    parser.add_argument('--model', choices=MODELS, default='small')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    try:
        run(parser.parse_args(argv))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
