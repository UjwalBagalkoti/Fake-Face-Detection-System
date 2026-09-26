from __future__ import annotations
import os
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torchvision import transforms
from .face_detector import FaceDetector
from .preprocess import crop_faces
from .model import ForensicFusion
class Predictor:
    def __init__(self,checkpoint=None,device=None,pretrained=False):
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); payload=torch.load(checkpoint,map_location=self.device) if checkpoint and os.path.exists(checkpoint) else None
        self.model=ForensicFusion(backbone=(payload or {}).get("backbone","efficientnet_b4"),pretrained=pretrained,temporal_frames=int((payload or {}).get("temporal_frames",24))); self.threshold=float((payload or {}).get("threshold",.5))
        if payload is not None: self.model.load_state_dict(payload.get("model",payload),strict=False)
        self.model.to(self.device).eval(); self.face_detector=FaceDetector(); self.tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def _tensor(self,bgr):
        return self.tf(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
    @torch.no_grad()
    def predict_face(self,face):
        p=torch.sigmoid(self.model(self._tensor(face))["logit"])[0].item(); return {"fake_probability":p,"label":"FAKE" if p>=self.threshold else "REAL"}
    def predict_image(self,bgr):
        faces=crop_faces(bgr,self.face_detector)
        if not faces: return self.predict_face(bgr)|{"faces_detected":0,"mode":"whole_image_fallback"}
        preds=[self.predict_face(f.image) for f in faces]; best=max(range(len(preds)),key=lambda i:preds[i]["fake_probability"])
        return preds[best]|{"faces_detected":len(faces),"face_scores":preds,"mode":"face_crop"}
    def predict_video(self,path,max_frames=24):
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): raise ValueError("Could not open video")
        frames=[]; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=cap.get(cv2.CAP_PROP_FPS) or 25.; ids=set(np.linspace(0,max(total-1,0),num=min(max_frames,max(total,1)),dtype=int).tolist()); i=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if i in ids:
                faces=crop_faces(frame,self.face_detector)
                if faces: frames.append(max(faces,key=lambda f:f.score).image)
            i+=1
        cap.release()
        if not frames: raise ValueError("No detectable faces found in video")
        with torch.no_grad():
            enc=self.model.encode_image(torch.cat([self._tensor(f) for f in frames],0))[0]; tp=torch.sigmoid(self.model.forward_sequence_embeddings(enc.unsqueeze(0)))[0].item(); fp=torch.sigmoid(self.model.image_head(enc).squeeze(1)).cpu().tolist()
        return {"label":"FAKE" if tp>=self.threshold else "REAL","fake_probability":tp,"frames_analyzed":len(frames),"fps":fps,"frame_probabilities":fp,"temporal_agreement":float(np.mean(np.array(fp)>=self.threshold))}
