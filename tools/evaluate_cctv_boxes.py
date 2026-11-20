"""Evaluate CAVIAR box tracking; these scores are not pixel-mask accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def box_iou(a, b):
    if a is None:
        return 0.0
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod()+np.maximum(0, b[2:]-b[:2]).prod()-inter
    return float(inter / max(1e-12, union))


def evaluate(masks, xml_path, mapping, start_frame=160):
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError('start_frame must be nonnegative')
    if not mapping or len(mapping) > 3 or len(set(mapping.values())) != len(mapping):
        raise ValueError('Use one to three unique target-to-reference identities')
    if xml_path.stat().st_size > 20*1024*1024:
        raise ValueError('Reference XML exceeds the evaluation size limit')
    reference = {}
    for frame in ET.parse(xml_path).getroot().findall('frame'):
        index = int(frame.attrib['number'])
        if index in reference:
            raise ValueError('Duplicate reference frame')
        reference[index] = {}
        for obj in frame.findall('./objectlist/object'):
            if obj.findtext('appearance', '').strip() != 'visible':
                continue
            b = obj.find('box').attrib
            x, y, w, h = [float(b[key]) for key in ('xc', 'yc', 'w', 'h')]
            reference[index][int(obj.attrib['id'])] = [x-w/2, y-h/2, x+w/2, y+h/2]
    rows = []
    digest = hashlib.sha256()
    for index, people in sorted(reference.items()):
        if index < start_frame:
            continue
        for uid, sid in sorted(mapping.items()):
            if sid not in people:
                continue
            path = masks / f'object_{uid}/{index:06d}.png'
            raw = path.read_bytes()
            mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2 or mask.dtype != np.uint8 or not set(np.unique(mask)) <= {0, 255}:
                raise ValueError(f'Invalid binary mask: {path.name}')
            digest.update(f'{uid}:{index}:'.encode()+hashlib.sha256(raw).digest())
            yy, xx = np.nonzero(mask)
            box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if len(xx) else None
            height, width = mask.shape
            target = np.clip(people[sid], [0, 0, 0, 0], [width, height, width, height])
            own = box_iou(box, target)
            other = max((box_iou(box, np.clip(b, [0, 0, 0, 0], [width, height, width, height])) for oid, b in people.items() if oid != sid), default=0.)
            rows.append({'frame': index, 'object_id': uid, 'bbox_iou': own, 'box_hit': own >= .5,
                         'empty': box is None, 'other_box_proxy': other > own and other >= .1})
    if not rows:
        raise ValueError('No visible reference observations in the selected range')

    def summarize(group):
        return {'visible_observations': len(group), 'mean_bbox_iou': float(np.mean([r['bbox_iou'] for r in group])),
                'box_hits_IoU05': sum(r['box_hit'] for r in group),
                'box_hit_rate': sum(r['box_hit'] for r in group)/len(group),
                'visible_empty_outputs': sum(r['empty'] for r in group),
                'other_box_proxy_NOT_confirmed_IDswitches': sum(r['other_box_proxy'] for r in group)}

    return {'schema_version': 1, 'metric': 'fixed-identity mask-bbox IoU, threshold 0.5; not pixel IoU/Dice',
            'reference_xml_sha256': hashlib.sha256(xml_path.read_bytes()).hexdigest(),
            'evaluated_mask_digest': digest.hexdigest(), 'start_frame': start_frame,
            'summary': summarize(rows),
            'by_object': {str(uid): summarize([r for r in rows if r['object_id'] == uid]) for uid in mapping if any(r['object_id'] == uid for r in rows)},
            'low_match_frames': [r for r in rows if not r['box_hit']],
            'limitation': 'Other-box proxy is not a confirmed identity-switch count; pixel GT is not used.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--object-map', nargs='+', default=['1:0', '2:1', '3:2'], help='outputID:XMLID')
    parser.add_argument('--start-frame', type=int, default=160)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    try:
        pairs = [tuple(map(int, item.split(':'))) for item in args.object_map]
        if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
            raise ValueError('Invalid or duplicate object mapping')
        mapping = dict(pairs)
        if any(uid <= 0 or sid < 0 for uid, sid in mapping.items()):
            raise ValueError('IDs must be positive output IDs and nonnegative reference IDs')
        result = evaluate(args.masks, args.reference, mapping, args.start_frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result['summary']))
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
