"""Local optical-flow support for SAM mask association, never output masks."""

import cv2
import numpy as np


def appearance_histogram(image, mask):
    """Small HSV descriptor; fewer than four pixels cannot support this check."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return None
    left, top, right, bottom = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    selected = mask[top:bottom, left:right].astype(np.uint8)*255
    histogram = cv2.calcHist([hsv], [0, 1], selected, [16, 16], [0, 180, 0, 256])
    histogram /= max(float(histogram.sum()), 1.)
    return histogram


def appearance_matches(reference, image, mask, limit=.6):
    histogram = appearance_histogram(image, mask)
    return reference is not None and histogram is not None and cv2.compareHist(
        reference, histogram, cv2.HISTCMP_BHATTACHARYYA) <= limit


def same_appearance(before, after, previous, candidate):
    return appearance_matches(appearance_histogram(before, previous), after, candidate, .35)


def motion_support(before, after, mask, context):
    """Move the last accepted mask using checked target-local feature tracks.

    Return None when motion is not supported. Callers retain their conservative
    stationary checks in that case. This estimate only guides association; it
    must never be exported as a predicted segmentation.
    """
    if before.shape != after.shape or before.shape[:2] != mask.shape:
        raise ValueError("Motion frames and target dimensions must match")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    padding = max(32, int(context // 2))
    left, top = max(0, int(xs.min())-padding), max(0, int(ys.min())-padding)
    right = min(width, int(xs.max())+padding+1)
    bottom = min(height, int(ys.max())+padding+1)
    first = cv2.cvtColor(before[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(after[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    local_mask = mask[top:bottom, left:right].astype(np.uint8)*255
    points = cv2.goodFeaturesToTrack(first, maxCorners=80, qualityLevel=.005,
                                    minDistance=2, mask=local_mask, blockSize=3)
    if points is None or len(points) < 4:
        return None
    options = dict(winSize=(31, 31), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    moved, forward, error = cv2.calcOpticalFlowPyrLK(first, second, points, None, **options)
    if moved is None or forward is None:
        return None
    back, backward, _ = cv2.calcOpticalFlowPyrLK(second, first, moved, None, **options)
    if back is None or backward is None:
        return None
    old, new = points[:, 0], moved[:, 0]
    valid = (forward[:, 0] > 0) & (backward[:, 0] > 0)
    valid &= np.isfinite(new).all(axis=1) & np.isfinite(back[:, 0]).all(axis=1)
    valid &= np.linalg.norm(back[:, 0]-old, axis=1) <= 1.5
    valid &= (new[:, 0] >= 0) & (new[:, 0] < first.shape[1])
    valid &= (new[:, 1] >= 0) & (new[:, 1] < first.shape[0])
    if error is not None:
        valid &= np.isfinite(error[:, 0]) & (error[:, 0] < 30)
    if int(valid.sum()) < 4:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(
        old[valid], new[valid], method=cv2.RANSAC,
        ransacReprojThreshold=2, maxIters=1000, confidence=.99)
    if transform is None or inliers is None or not np.isfinite(transform).all():
        return None
    if int(inliers.sum()) < 4 or float(inliers.mean()) < .6:
        return None
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    if not .7 <= scale <= 1.4:
        return None
    center = np.array([xs.mean()-left, ys.mean()-top])
    predicted = transform[:, :2] @ center + transform[:, 2]
    if np.linalg.norm(predicted-center) > context/3:
        return None
    # Convert the local transform back to full-image coordinates.
    origin = np.array([left, top], dtype=np.float64)
    transform[:, 2] += origin-transform[:, :2] @ origin
    aligned = cv2.warpAffine(mask.astype(np.uint8), transform, (width, height),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    return aligned if aligned.any() else None
