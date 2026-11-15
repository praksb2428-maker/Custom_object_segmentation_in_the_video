"""Bounded reproducible benchmark; synthetic mask accuracy and real samples stay separate."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=float,required=True)
    parser.add_argument('--truth',type=Path)
    parser.add_argument('--stage',type=int,default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0,str(args.project.resolve()))
    import cv2
    import numpy as np
    import faulthandler
    faulthandler.dump_traceback_later(120, exit=True)
    import torch
    faulthandler.cancel_dump_traceback_later()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    from segmentation.prompts import load_prompts
    from segmentation.tracker import Sam2Tracker
    from segmentation.continuous import ContinuousTracker
    from segmentation.masks import render_frame
    from segmentation.video import read_image,write_image
    files = sorted(args.frames.glob('*.jpg'))
    if not files:
        raise ValueError('No frames')
    first = read_image(files[0]); height,width = first.shape[:2]
    prompts = load_prompts(args.prompts,width,height)
    truth = np.load(args.truth)['labels'] if args.truth else None
    if truth is not None and truth.shape != (len(files),height,width):
        raise ValueError('Ground truth dimensions differ')
    (args.output/'candidate-identity.json').write_text(json.dumps({'schema_version':1,'stage':args.stage,'files':{str(p.relative_to(args.project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.project/'segmentation').glob('*.py'))},'prompts_sha256':hashlib.sha256(args.prompts.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'frame_count':len(files),'size':[width,height],'fps':args.fps,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)},ensure_ascii=False,indent=2),encoding='utf-8')
    started=time.perf_counter()
    adapter=Sam2Tracker(args.checkpoint,'small','cuda')
    tracker=ContinuousTracker(adapter)
    if hasattr(tracker, "adaptive_identity"):
        tracker.adaptive_identity = args.stage >= 1
    if args.stage >= 2:
        tracker.enable_reacquisition=True
    if args.stage >= 3:
        tracker.enable_small_refinement=True
    if hasattr(tracker, "enable_prompt_refinement"):
        tracker.enable_prompt_refinement = args.stage >= 4
    load_seconds=time.perf_counter()-started
    stats={p.object_id:{'visible_ious':[],'visible_dices':[],'hidden_false_positives':0,'hidden_frames':0,'nonempty':0} for p in prompts}
    writer=cv2.VideoWriter(str(args.output/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot open output video')
    trajectories={}
    samples=set([0,12,24,29,30,60,89,98,99,105,120,143]) if truth is not None else set([0,120,165,185,334,385,386,445,480,528,575])
    generator=tracker.predict(args.frames,prompts)
    started=time.perf_counter();seen=0
    status_source=args.frames.parent/'tracking_status.csv'
    try:
        for i,masks in generator:
            if i!=seen or set(masks)!={p.object_id for p in prompts}:raise RuntimeError('Unexpected frames or IDs')
            frame=read_image(files[i]);_,overlay=render_frame(frame,masks,trajectories,True)
            writer.write(overlay)
            for oid,mask in masks.items():
                stats[oid]['nonempty']+=int(mask.any())
                if truth is not None:
                    gt=truth[i]==oid
                    if gt.any():
                        intersection=int((gt&mask).sum());union=int((gt|mask).sum())
                        stats[oid]['visible_ious'].append(intersection/union)
                        stats[oid]['visible_dices'].append(2*intersection/(int(gt.sum())+int(mask.sum())))
                    else:
                        stats[oid]['hidden_frames']+=1
                        stats[oid]['hidden_false_positives']+=int(mask.any())
                if i in samples:
                    dest=args.output/'samples'/f'object_{oid}';dest.mkdir(parents=True,exist_ok=True)
                    write_image(dest/f'{i:06d}.png',mask.astype(np.uint8)*255)
            if i in samples:
                write_image(args.output/f'frame-{i:06d}.png',overlay)
            seen+=1
    finally:
        if hasattr(generator,'close'):generator.close()
        writer.release()
    elapsed=time.perf_counter()-started
    if seen!=len(files):raise RuntimeError('Incomplete benchmark')
    if status_source.exists():
        # Keep only compact run evidence; next benchmark will rewrite the shared status path.
        status=(args.output/'tracking_status.csv');status.write_bytes(status_source.read_bytes())
        with status.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        state_counts={str(p.object_id):dict(Counter(r['state'] for r in rows if int(r['object_id'])==p.object_id)) for p in prompts}
    else:state_counts={}
    refiner=getattr(tracker,'small_refiner',None)
    prompt_refiner=getattr(tracker,'prompt_refiner',None)
    summary={'features':{'adaptive_identity':getattr(tracker,'adaptive_identity',False),'reacquisition':getattr(tracker,'enable_reacquisition',False),'small_refinement':getattr(tracker,'enable_small_refinement',False),'prompt_refinement':getattr(tracker,'enable_prompt_refinement',False),'prompt_calls':getattr(prompt_refiner,'calls',0),'prompt_accepted':getattr(prompt_refiner,'accepted',0),'prompt_applied':getattr(prompt_refiner,'applied',0),'refinement_calls':getattr(refiner,'calls',0),'refinement_accepted':getattr(refiner,'accepted',0)},'schema_version':1,'stage':args.stage,'frame_count':seen,'model_load_seconds':load_seconds,'benchmark_seconds':elapsed,'fps_including_overlay_export':seen/elapsed,'ground_truth':'deterministic synthetic visible masks' if truth is not None else None,'real_accuracy':'sampled visual/point review required; tracked is not accuracy','state_counts':state_counts,'objects':{str(oid):{'mean_visible_iou':float(np.mean(s['visible_ious'])) if s['visible_ious'] else None,'mean_visible_dice':float(np.mean(s['visible_dices'])) if s['visible_dices'] else None,'hidden_false_positive_frames':s['hidden_false_positives'],'hidden_frames':s['hidden_frames'],'nonempty_frames':s['nonempty']} for oid,s in stats.items()}}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()