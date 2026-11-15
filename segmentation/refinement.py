"""Conservative SAM crop refinement of already visible, small object masks."""
import numpy as np
import cv2
from .focused import crop_bounds, local_prompt, mask_prompt


class SmallObjectRefiner:
    def __init__(self, adapter, max_extent=64, max_area=1024):
        self.adapter = adapter
        self.max_extent, self.max_area = max_extent, max_area
        self.calls = self.accepted = 0

    def refine(self, image, mask):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return mask
        extent = max(int(np.ptp(xs))+1, int(np.ptp(ys))+1)
        if extent > self.max_extent or len(xs) > self.max_area:
            return mask
        height, width = mask.shape
        bounds = crop_bounds((float(xs.mean()), float(ys.mean())), max(64, extent*4), width, height)
        left, top, right, bottom = bounds
        prompt = local_prompt(mask_prompt(1, mask), bounds)
        self.calls += 1
        candidate, score = self.adapter.refine_image(image[top:bottom, left:right], prompt)
        candidate = np.asarray(candidate, dtype=bool)
        if candidate.shape != (bottom-top, right-left):
            raise RuntimeError("Refined mask dimensions do not match the crop")
        if not np.isfinite(score) or score < .8 or not candidate.any():
            return mask
        original = mask[top:bottom, left:right]
        area_ratio = int(candidate.sum())/max(int(original.sum()), 1)
        iou = int((candidate & original).sum())/max(int((candidate | original).sum()), 1)
        # A crop boundary is often an unsegmented background proposal.
        if not .5 <= area_ratio <= 2 or iou < .5 or (candidate[0].any() or candidate[-1].any()
                or candidate[:, 0].any() or candidate[:, -1].any()):
            return mask
        result = np.zeros_like(mask)
        result[top:bottom, left:right] = candidate
        self.accepted += 1
        return result


class ReferenceMaskRefiner:
    """Guarded crop re-segmentation of an actual last-good reference frame."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.calls = self.accepted = self.applied = 0

    def refine(self, image, prompt, raw, prompts, raw_masks):
        from .prompts import ObjectPrompt
        from .focused import prompt_center
        ys, xs = np.nonzero(raw)
        if prompt.box is None or not len(xs):
            return raw
        if len(xs) <= 1024 and max(int(np.ptp(xs))+1, int(np.ptp(ys))+1) <= 64:
            return raw  # Already covered by small-object refinement.
        height, width = raw.shape
        x1,y1,x2,y2 = prompt.box
        bounds = crop_bounds(((x1+x2)/2, (y1+y2)/2),
                             max(256, int(np.ceil(max(x2-x1,y2-y1)*2))), width, height)
        left, top, right, bottom = bounds
        points, labels = list(prompt.points), list(prompt.labels)
        for other in prompts:
            if other.object_id == prompt.object_id:
                continue
            px, py = prompt_center(other)
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                points.append([px,py]); labels.append(0)
        local = local_prompt(ObjectPrompt(prompt.object_id, prompt.box, points, labels), bounds)
        self.calls += 1
        candidate, score = self.adapter.refine_image(image[top:bottom,left:right], local)
        candidate = np.asarray(candidate, dtype=bool)
        if candidate.shape != (bottom-top,right-left):
            raise RuntimeError("Reference refinement dimensions do not match the crop")
        if not np.isfinite(score) or score < .8 or not candidate.any():
            return raw
        result = np.zeros_like(raw);result[top:bottom,left:right] = candidate
        area = int(result.sum());old_area = len(xs)
        coverage = int((result & raw).sum())/old_area
        _, _, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8))
        major_parts = int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA] >= old_area*.1))
        # A head/leg-only seed can have far less area than the same object's
        # complete visible body. Permit that expansion only with strong support.
        upper_ratio = 8 if major_parts >= 2 and score >= .9 and coverage >= .9 else 4
        if not .5 <= area/old_area <= upper_ratio or coverage < .65:
            return raw
        # The selected box must contain most of this object. It is not permission
        # to turn a crop-wide background proposal into an object mask.
        ix1,iy1 = max(0,int(x1)),max(0,int(y1))
        ix2,iy2 = min(width,int(np.ceil(x2))),min(height,int(np.ceil(y2)))
        if int(result[iy1:iy2,ix1:ix2].sum())/area < .5:
            return raw
        if candidate[0].any() or candidate[-1].any() or candidate[:,0].any() or candidate[:,-1].any():
            return raw
        for (px,py),label in zip(local.points,local.labels):
            x,y=int(round(px)),int(round(py))
            if 0<=x<candidate.shape[1] and 0<=y<candidate.shape[0] and bool(candidate[y,x]) != bool(label):
                return raw
        for oid, rival in raw_masks.items():
            if oid == prompt.object_id:
                continue
            rival = np.asarray(rival,dtype=bool)
            if rival.shape != raw.shape:
                raise RuntimeError("Reference rival mask dimensions do not match the video")
            if rival.any() and int((result&rival).sum())/min(area,int(rival.sum())) > .35:
                return raw
        self.accepted += 1
        return result


def reference_prompt(object_id, mask):
    """A box can recover missing interior parts that fragment-only points omit."""
    from .prompts import ObjectPrompt
    ys,xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot refine an empty reference mask")
    height,width = mask.shape
    extent = max(int(np.ptp(xs))+1,int(np.ptp(ys))+1)
    horizontal = extent*.35
    vertical = extent*.05
    return ObjectPrompt(object_id,[max(0,float(xs.min())-horizontal),
                                  max(0,float(ys.min())-vertical),
                                  min(width,float(xs.max()+1)+horizontal),
                                  min(height,float(ys.max()+1)+vertical)],[],[])
