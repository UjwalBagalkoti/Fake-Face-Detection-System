from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from detection.model import ForensicFusion
from training.datasets import build_imagefolder
from training.metrics import binary_metrics,save_metrics
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--model",required=True); ap.add_argument("--split",default="test",choices=["train","val","test"]); ap.add_argument("--batch-size",type=int,default=32); a=ap.parse_args()
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck=torch.load(a.model,map_location=d); m=ForensicFusion(backbone=ck.get("backbone","efficientnet_b4"),pretrained=False).to(d); m.load_state_dict(ck.get("model",ck),strict=False); m.eval()
    dl=DataLoader(build_imagefolder(Path(a.data)/a.split),batch_size=a.batch_size); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl: ps+=torch.sigmoid(m(x.to(d))["logit"]).cpu().tolist(); ys+=y.tolist()
    met=binary_metrics(ys,ps,ck.get("threshold",.5)); print(met); save_metrics(met,f"models/{a.split}_metrics.json")
if __name__=="__main__": main()
