from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from detection.predictor import Predictor
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--model",default=None); a=ap.parse_args()
    p=Predictor(a.model); ext=Path(a.path).suffix.lower()
    result=p.predict_video(a.path) if ext in {".mp4",".mov",".avi",".mkv",".webm"} else p.predict_image(cv2.imread(a.path))
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
