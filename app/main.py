from __future__ import annotations
import os,tempfile,logging,traceback
from pathlib import Path
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from detection.predictor import Predictor

app=FastAPI(title="Fake Face Detection System",version="1.2.0")
BASE=Path(__file__).resolve().parent.parent
STATIC=BASE/"static"
IMAGE_MODEL=os.environ.get("FORENSIC_MODEL",str(BASE/"models"/"best_image_model.pth"))
VIDEO_MODEL=os.environ.get("FORENSIC_VIDEO_MODEL",str(BASE/"models"/"best_video_model.pth"))
predictor=None

def get_predictor():
    global predictor
    if predictor is None:
        if Path(IMAGE_MODEL).exists():
            predictor=Predictor(image_checkpoint=IMAGE_MODEL,video_checkpoint=VIDEO_MODEL if Path(VIDEO_MODEL).exists() else None,pretrained=False)
        else:
            predictor=Predictor(image_checkpoint=None,video_checkpoint=None,pretrained=False)
    return predictor

@app.get("/")
def home(): return FileResponse(STATIC/"index.html")

@app.get("/health")
def health():
    p=get_predictor()
    return {
        "status":"ok","device":str(p.device),
        "model_source":p.model_source,
        "image_model":IMAGE_MODEL if Path(IMAGE_MODEL).exists() else None,
        "video_model":VIDEO_MODEL if Path(VIDEO_MODEL).exists() else None,
        "video_temporal_loaded":p.video_model is not None
    }

@app.post("/api/predict/image")
async def predict_image(file:UploadFile=File(...)):
    if not (file.content_type or "").startswith("image/"): raise HTTPException(400,"Upload an image file")
    raw=await file.read()
    if len(raw)>15_000_000: raise HTTPException(413,"Image is too large")
    import cv2,numpy as np
    img=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,"Invalid image")
    try:
        return get_predictor().predict_image(img)
    except Exception as e:
        logging.exception("IMAGE PREDICTION FAILED")
        raise HTTPException(500,f"Prediction failed: {type(e).__name__}: {e}") from e

@app.post("/api/predict/video")
async def predict_video(file:UploadFile=File(...)):
    if file.content_type not in {"video/mp4","video/quicktime","video/webm","video/x-msvideo"}:
        raise HTTPException(400,"Upload MP4, MOV, WebM or AVI")
    raw=await file.read()
    if len(raw)>200_000_000: raise HTTPException(413,"Video is too large")
    suffix=Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f: f.write(raw); temp=f.name
    try:
        try:
            return get_predictor().predict_video(temp)
        except Exception as e:
            logging.exception("VIDEO PREDICTION FAILED")
            raise HTTPException(500,f"Prediction failed: {type(e).__name__}: {e}") from e
    finally: Path(temp).unlink(missing_ok=True)

app.mount("/static",StaticFiles(directory=STATIC),name="static")
