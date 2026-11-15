"""Limited source-only proxy metrics, explicitly not independent pixel ground truth."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image


def evaluate(maskroot, reference_root, frames):
    sparse=json.loads((reference_root/'sparse-points.json').read_text(encoding='utf-8'))
    extra=json.loads((reference_root/'additional-points.json').read_text(encoding='utf-8'))
    positives={int(f):{oid:list(pts) for oid,pts in objects.items()} for f,objects in sparse['frames'].items()}
    for f,objects in extra['additional_positive'].items():
        positives[int(f)]={oid:[[x*3,y*3] for x,y in pts] for oid,pts in objects.items()}
    exclusion_path=reference_root/'point-exclusions.json'
    exclusions=json.loads(exclusion_path.read_text(encoding='utf-8'))['background'] if exclusion_path.exists() else []
    excluded={(item['frame'],str(item['object_id']),tuple(item['point'])) for item in exclusions}
    rows={}
    for i in frames:
        for oid,points in positives[i].items():
            row=rows.setdefault(oid,{'positive_hits':0,'positive_points':0,'existing_sparse_hits':0,'existing_sparse_points':0,'additional_hits':0,'additional_points':0,'background_hits':0,'background_points':0,'other_object_hits':0,'other_object_points':0,'approximate_ious':[]})
            path=maskroot/f'object_{oid}'/f'{i:06d}.png'
            if not path.exists():raise FileNotFoundError(path)
            mask=np.array(Image.open(path))>0
            assert mask.shape==(1080,1920)
            for x,y in points:
                hit=int(mask[y,x]);row['positive_hits']+=hit;row['positive_points']+=1
                scope='existing_sparse' if str(i) in sparse['frames'] else 'additional'
                row[scope+'_hits']+=hit;row[scope+'_points']+=1
            for x,y in extra['background'][str(i)][oid]:
                if (i,oid,(x,y)) in excluded:continue
                row['background_hits']+=int(mask[y*3,x*3]);row['background_points']+=1
            for other,pts in positives[i].items():
                if other==oid:continue
                for x,y in pts:row['other_object_hits']+=int(mask[y,x]);row['other_object_points']+=1
            proxy=reference_root/'proxy-masks'/f'object_{oid}'/f'{i:06d}.png'
            if proxy.exists():
                gt=np.array(Image.open(proxy))>0
                row['approximate_ious'].append({'frame':i,'iou':int((gt&mask).sum())/max(1,int((gt|mask).sum()))})
    for row in rows.values():
        row['mean_approximate_iou']=float(np.mean([x['iou'] for x in row['approximate_ious']])) if row['approximate_ious'] else None
    return rows


def compare(baseline,candidate):
    reasons=[]
    for oid,a in baseline.items():
        b=candidate[oid]
        for key in ['background_hits','other_object_hits']:
            if b[key]>a[key]:reasons.append(f'object {oid}: increased {key}')
        if b['existing_sparse_hits']<a['existing_sparse_hits']:reasons.append(f'object {oid}: fewer original sparse point hits')
        if b['positive_hits']<a['positive_hits']:reasons.append(f'object {oid}: fewer interior point hits')
        if b['mean_approximate_iou']<a['mean_approximate_iou']-.02:reasons.append(f'object {oid}: proxy IoU regression')
    old=float(np.mean([v['mean_approximate_iou'] for v in baseline.values()]));new=float(np.mean([v['mean_approximate_iou'] for v in candidate.values()]))
    if new<old+.05:reasons.append('proxy mean IoU gain below .05')
    if candidate['2']['existing_sparse_hits']<=baseline['2']['existing_sparse_hits']:reasons.append('object2 original sparse coverage did not improve')
    return {'pass':not reasons,'findings':reasons,'mean_approximate_iou_before':old,'mean_approximate_iou_after':new,'limits':'Coarse assistant body-coverage polygons on2frames and fixed point probes;575ID1 head outline is approximate. Not precise boundaries, full-video or independent human pixel ground truth.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--frames',default='0,120,185,334,575');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    frames=[int(f) for f in args.frames.split(',')]
    a=evaluate(args.baseline,args.references,frames);b=evaluate(args.candidate,args.references,frames)
    result={'schema_version':1,'frames':frames,'baseline':a,'candidate':b,'comparison':compare(a,b)}
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
