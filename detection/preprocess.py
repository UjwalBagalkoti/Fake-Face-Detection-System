from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np
@dataclass
class FaceCrop:
    image: np.ndarray
    box: tuple[int,int,int,int]
    score: float
def expand_box(box,width,height,margin=0.20):
    x1,y1,x2,y2=map(int,box); w=x2-x1; h=y2-y1; dx=int(w*margin); dy=int(h*margin)
    return max(0,x1-dx),max(0,y1-dy),min(width,x2+dx),min(height,y2+dy)
def crop_faces(frame,detector,margin=0.20):
    h,w=frame.shape[:2]; out=[]
    for box,score in detector.detect(frame):
        x1,y1,x2,y2=expand_box(box,w,h,margin); crop=frame[y1:y2,x1:x2]
        if crop.size: out.append(FaceCrop(crop,(x1,y1,x2,y2),float(score)))
    return out
def read_image_bytes(raw):
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Could not decode image")
    return image
