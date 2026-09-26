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
    def __init__(self,image_checkpoint=None,video_checkpoint=None,device=None,pretrained=False):
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_payload=self._load_payload(image_checkpoint)
        self.video_payload=self._load_payload(video_checkpoint)
        payload=self.image_payload or self.video_payload or {}
        backbone=payload.get("backbone","efficientnet_b4")
        temporal_frames=int(payload.get("temporal_frames",24))
        self.model=ForensicFusion(backbone=backbone,pretrained=pretrained,temporal_frames=temporal_frames)
        if self.image_payload is not None:
            self.model.load_state_dict(self.image_payload.get("model",self.image_payload),strict=False)
        self.image_threshold=float(self.image_payload.get("threshold",.5) if self.image_payload else .5)
        if self.video_payload is not None:
            self.video_model=ForensicFusion(backbone=backbone,pretrained=False,temporal_frames=int(self.video_payload.get("temporal_frames",24)))
            self.video_model.load_state_dict(self.video_payload.get("model",self.video_payload),strict=False)
            self.video_model.to(self.device).eval()
            self.video_threshold=float(self.video_payload.get("threshold",self.image_threshold))
        else:
            self.video_model=None; self.video_threshold=self.image_threshold
        self.model.to(self.device).eval()
        self.face_detector=FaceDetector()
        self.tf=transforms.Compose([
            transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        ])

    @staticmethod
    def _load_payload(path):
        return torch.load(path,map_location="cpu") if path and os.path.exists(path) else None

    def _tensor(self,bgr):
        return self.tf(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)

    def _label(self,p,threshold):
        margin=0.05
        if abs(float(p)-threshold)<margin:
            return "UNCERTAIN"
        return "FAKE" if p>=threshold else "REAL"

    @torch.no_grad()
    def predict_face(self,face,tta=True):
        x=self._tensor(face)
        prob=torch.sigmoid(self.model(x)["logit"])[0].item()
        probs=[prob]
        if tta:
            flipped=cv2.flip(face,1)
            probs.append(torch.sigmoid(self.model(self._tensor(flipped))["logit"])[0].item())
        p=float(np.mean(probs))
        gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        quality="low" if blur<40 or min(face.shape[:2])<96 else "ok"
        return {
            "fake_probability":p,
            "label":self._label(p,self.image_threshold),
            "tta_samples":len(probs),
            "face_quality":quality,
            "blur_score":blur
        }

    def predict_image(self,bgr):
        faces=crop_faces(bgr,self.face_detector)
        if not faces:
            return self.predict_face(bgr)|{"faces_detected":0,"mode":"whole_image_fallback"}
        preds=[self.predict_face(f.image) for f in faces]
        probs=[x["fake_probability"] for x in preds]
        max_i=int(np.argmax(probs))
        return {
            "label":"FAKE" if probs[max_i]>=self.image_threshold else "REAL",
            "fake_probability":float(probs[max_i]),
            "faces_detected":len(faces),
            "face_scores":preds,
            "mean_fake_probability":float(np.mean(probs)),
            "mode":"face_crop"
        }

    def predict_video(self,path,max_frames=24):
        model=self.video_model or self.model
        threshold=self.video_threshold
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): raise ValueError("Could not open video")
        frames=[]; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=cap.get(cv2.CAP_PROP_FPS) or 25.
        sample_count=min(max_frames,max(total,1))
        ids=set(np.linspace(0,max(total-1,0),num=sample_count,dtype=int).tolist())
        i=0
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
            batch=torch.cat([self._tensor(f) for f in frames],0)
            enc,fmap=model.encode_image(batch)
            frame_probs=torch.sigmoid(model.image_head(enc).squeeze(1)).cpu().numpy()
            if self.video_model is not None:
                n=min(model.temporal_frames,len(frames))
                seq=enc[:n].unsqueeze(0)
                temporal_prob=torch.sigmoid(model.forward_sequence_embeddings(seq))[0].item()
                fused=float(.65*temporal_prob+.35*float(np.mean(frame_probs)))
            else:
                temporal_prob=None
                fused=float(np.mean(frame_probs))
        return {
            "label":"FAKE" if fused>=threshold else "REAL",
            "fake_probability":fused,
            "frames_analyzed":len(frames),
            "fps":float(fps),
            "frame_probabilities":[float(x) for x in frame_probs],
            "temporal_probability":temporal_prob,
            "temporal_model_loaded":self.video_model is not None,
            "temporal_agreement":float(np.mean(frame_probs>=threshold)),
            "mode":"video_temporal_fusion" if self.video_model is not None else "video_frame_aggregation"
        }
