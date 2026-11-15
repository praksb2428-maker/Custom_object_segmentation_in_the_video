"""Create a deterministic synthetic three-object segmentation benchmark."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def generate(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    frames = root / 'frames'
    frames.mkdir(parents=True)
    width, height, count, fps = 640, 360, 144, 24
    rng = np.random.default_rng(20261115)
    base = rng.integers(42, 69, (height, width, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (7, 7), 0)
    for y in range(0, height, 30):
        cv2.line(base, (0, y), (width-1, y), (75, 72, 64), 1)
    truth = []
    first_boxes = {}
    writer = cv2.VideoWriter(str(root/'synthetic.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot write benchmark video')
    for i in range(count):
        image = base.copy()
        label = np.zeros((height, width), np.uint8)
        positions = {1: (60+2*i, 70+int(8*np.sin(i/15))), 2: (360+i, 170+int(4*np.sin(i/12))), 3: (590-int(1.5*i), 275+int(7*np.sin(i/10)))}
        for oid, (x, y) in positions.items():
            mask = np.zeros((height, width), np.uint8)
            if oid == 1:
                cv2.circle(mask, (x, y), 10, 1, -1)
                fade = max(0, 1-abs(i-62)/45)*.65
                color = np.array([40, 60, 220])*(1-fade)+np.array([150,150,150])*fade
            elif oid == 2:
                cv2.rectangle(mask, (x-10,y-8), (x+10,y+8), 1, -1)
                color = np.array([220, 110, 35])
            else:
                size = int(round(11+5*np.sin(np.pi*min(i,110)/110))) if i < 80 else max(5, 15-(i-80)//6)
                cv2.fillPoly(mask,[np.array([[x,y-size],[x-size,y+size],[x+size,y+size]])],1)
                color = np.array([45, 205, 75])
            ys, xs = np.nonzero(mask)
            texture = ((xs-x)//3+(ys-y)//3)%2
            values = np.clip(color[None,:]+texture[:,None]*28,0,255).astype(np.uint8)
            image[ys,xs] = values
            label[mask>0] = oid
            if i == 0:
                first_boxes[oid] = [int(xs.min())-2,int(ys.min())-2,int(xs.max())+3,int(ys.max())+3]
        if 30 <= i <= 98:
            cv2.rectangle(image,(365,145),(485,195),(112,108,100),-1)
            cv2.rectangle(image,(365,145),(485,195),(155,150,140),2)
            label[145:196,365:486] = 0
        if 105 <= i <= 135:
            cv2.circle(image,(535,70),10,(40,60,220),-1)
        ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:
            raise RuntimeError('Cannot encode fixture')
        (frames/f'{i:06d}.jpg').write_bytes(encoded.tobytes())
        writer.write(image)
        truth.append(label)
    writer.release()
    np.savez_compressed(root/'truth.npz',labels=np.stack(truth))
    objects = [{'object_id':oid,'box':box,'points':[[float(positions[oid][0]),float(positions[oid][1])]],'labels':[1]} for oid,box in first_boxes.items()]
    # Positive points belong to the first frame, not the last loop iteration.
    for obj in objects:
        x1,y1,x2,y2 = obj['box']
        obj['points'] = [[(x1+x2)/2,(y1+y2)/2]]
    (root/'prompts.json').write_text(json.dumps({'schema_version':1,'frame_index':0,'width':width,'height':height,'objects':objects},indent=2),encoding='utf-8')
    (root/'fixture.json').write_text(json.dumps({'schema_version':1,'kind':'synthetic; not a natural-video accuracy claim','seed':20261115,'width':width,'height':height,'frame_count':count,'fps':fps,'occlusion':{'object_id':2,'first_frame':30,'last_frame':98},'conditions':['gradual desaturation','scale change','full occlusion longer than 60 frames','same-color distractor']},indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    generate(parser.parse_args().output)