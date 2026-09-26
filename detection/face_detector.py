from __future__ import annotations
import cv2
class FaceDetector:
    def __init__(self,min_face_size=80):
        self.cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        if self.cascade.empty(): raise RuntimeError("OpenCV face cascade could not be loaded")
        self.min_face_size=min_face_size
    def detect(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); h,w=frame.shape[:2]; area=float(max(1,h*w)); out=[]
        for x,y,fw,fh in self.cascade.detectMultiScale(gray,scaleFactor=1.08,minNeighbors=5,minSize=(self.min_face_size,self.min_face_size)):
            score=min(0.99,0.50+1.25*((fw*fh)/area)**0.5); out.append(((x,y,x+fw,y+fh),score))
        return out
