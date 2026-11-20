"""Fixed opt-in edge shrink eligibility; no model, reference labels or cache."""
import cv2
import numpy as np

def supported_edge_shrink(current, previous, *, consecutive):
    if not consecutive or previous is None:
        return False
    current=np.asarray(current,dtype=bool);previous=np.asarray(previous,dtype=bool)
    if current.ndim!=2 or current.shape!=previous.shape:
        return False
    area=int(current.sum());old_area=int(previous.sum())
    if area<4 or old_area<4 or not .125<=area/old_area<=1.:
        return False
    # Bands refer to the actual image, never a crop boundary. Fixed two pixels.
    def edges(mask):
        return (mask[:2].any(),mask[-2:].any(),mask[:,:2].any(),mask[:,-2:].any())
    if not any(a and b for a,b in zip(edges(current),edges(previous))):
        return False
    support=cv2.dilate(previous.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    return int((current&support).sum())>=area*.5
