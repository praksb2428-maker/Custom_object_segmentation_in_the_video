"""Compare fixed benchmark runs without treating tracker states as accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sparse_probe(folder, reference):
    result={}
    for frame,objects in reference['frames'].items():
        for oid,points in objects.items():
            row=result.setdefault(oid,{'covered':0,'positive_points':0,'other_object_hits':0,'other_object_points':0})
            mask=np.array(Image.open(folder/'samples'/f'object_{oid}'/f'{int(frame):06d}.png'))>0
            for x,y in points:
                row['covered']+=int(mask[y,x]);row['positive_points']+=1
            for other,other_points in objects.items():
                if other==oid:continue
                for x,y in other_points:
                    row['other_object_hits']+=int(mask[y,x]);row['other_object_points']+=1
    return result


def compare(root):
    reference=json.loads((root/'data/real/sparse-points.json').read_text(encoding='utf-8'))
    runs={}
    for name in ['baseline','part1','part2','part3']:
        if not all((root/name/k/'summary.json').exists() for k in ['synthetic','real']):continue
        synth=json.loads((root/name/'synthetic/summary.json').read_text(encoding='utf-8'))
        real=json.loads((root/name/'real/summary.json').read_text(encoding='utf-8'))
        runs[name]={'synthetic':synth,'real':real,'sparse_real_points':sparse_probe(root/name/'real',reference)}
    comparisons={}
    previous='baseline'
    for name in ['part1','part2','part3']:
        if name not in runs:continue
        a=runs[previous];b=runs[name]
        reasons=[]
        a_mean=np.mean([v['mean_visible_iou'] for v in a['synthetic']['objects'].values()])
        b_mean=np.mean([v['mean_visible_iou'] for v in b['synthetic']['objects'].values()])
        for oid,old in a['synthetic']['objects'].items():
            new=b['synthetic']['objects'][oid]
            if new['mean_visible_iou'] < old['mean_visible_iou']-.01:reasons.append(f'object {oid} synthetic IoU regression')
            if new['hidden_false_positive_frames'] > old['hidden_false_positive_frames']:reasons.append(f'object {oid} hidden false positive increase')
        if b_mean < a_mean:reasons.append('mean synthetic visible IoU decreased')
        for oid,old in a['sparse_real_points'].items():
            new=b['sparse_real_points'][oid]
            if new['covered']<old['covered']:reasons.append(f'object {oid} sparse real point coverage decreased')
            if new['other_object_hits']>old['other_object_hits']:reasons.append(f'object {oid} other-object point contamination increased')
        comparisons[name]={'against':previous,'quantitative_gate_pass':not reasons,'findings':reasons,'mean_synthetic_visible_iou_before':float(a_mean),'mean_synthetic_visible_iou_after':float(b_mean),'requires_sampled_visual_review':True,'real_pixelwise_iou':'not available; sparse approximate points only','timing_scope':'same local benchmark including overlay export; one run each, not statistical speed evidence'}
        previous=name
    return {'schema_version':1,'runs':runs,'comparisons':comparisons,'limits':['Synthetic geometry benchmark is not natural-video accuracy.','Real probes are 16 assistant-marked interior points across four frames, not full-mask ground truth.','Tracked/recovered counts are program states, not correctness.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    value=compare(args.root)
    (args.root/'comparison.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'comparisons':value['comparisons'],'real_points':{k:v['sparse_real_points'] for k,v in value['runs'].items()}},ensure_ascii=False,indent=2))