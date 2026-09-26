"""Evaluate a checkpoint on a separate dataset without recalibrating its threshold."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from detection.model import ForensicFusion
from training.datasets import build_imagefolder
from training.metrics import binary_metrics
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--model",required=True); ap.add_argument("--batch-size",type=int,default=32); a=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(a.model,map_location=device); model=ForensicFusion(backbone=ckpt.get("backbone","efficientnet_b4"),pretrained=False).to(device)
    model.load_state_dict(ckpt.get("model",ckpt),strict=False); model.eval()
    dl=DataLoader(build_imagefolder(Path(a.data),False),batch_size=a.batch_size,shuffle=False); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl: ps.extend(torch.sigmoid(model(x.to(device))["logit"]).cpu().tolist()); ys.extend(y.tolist())
    print(binary_metrics(ys,ps,float(ckpt.get("threshold",0.5))))
if __name__=="__main__": main()
