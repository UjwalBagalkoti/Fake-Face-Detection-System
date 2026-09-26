from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from detection.model import ForensicFusion
from training.datasets import eval_transform
from training.metrics import binary_metrics,calibrate_threshold,save_metrics

class VideoFolderDataset(Dataset):
    def __init__(self,root,frames=24):
        self.items=[]; self.frames=frames; self.tf=eval_transform(); root=Path(root)
        for name,label in [("real",0),("fake",1)]:
            for d in sorted((root/name).glob("*")):
                imgs=sorted([p for p in d.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
                if imgs:self.items.append((imgs,label,d.name))
        if not self.items: raise RuntimeError(f"No video frame folders found under {root}")
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        paths,label,name=self.items[i]
        ids=np.linspace(0,len(paths)-1,num=min(self.frames,len(paths)),dtype=int)
        frames=[self.tf(Image.open(paths[j]).convert("RGB")) for j in ids]
        while len(frames)<self.frames: frames.append(frames[-1].clone())
        return torch.stack(frames),torch.tensor(label,dtype=torch.float32),name

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--image-model",required=True)
    ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--finetune-image",action="store_true",help="also update spatial/frequency image features")
    a=ap.parse_args()
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.image_model,map_location=d)
    m=ForensicFusion(backbone=ck.get("backbone","efficientnet_b4"),pretrained=False).to(d)
    m.load_state_dict(ck.get("model",ck),strict=False)
    if not a.finetune_image:
        for module in [m.spatial,m.frequency,m.image_fuse,m.image_head]:
            for p in module.parameters(): p.requires_grad=False
    tr=DataLoader(VideoFolderDataset(Path(a.data)/"train"),batch_size=a.batch_size,shuffle=True)
    va=DataLoader(VideoFolderDataset(Path(a.data)/"val"),batch_size=a.batch_size)
    opt=optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr)
    loss=nn.BCEWithLogitsLoss(); best=-1
    for e in range(a.epochs):
        m.train()
        for frames,y,_ in tr:
            b,t,c,h,w=frames.shape
            with torch.set_grad_enabled(a.finetune_image):
                emb=m.encode_image(frames.to(d).reshape(b*t,c,h,w))[0].reshape(b,t,-1)
            z=m.forward_sequence_embeddings(emb); l=loss(z,y.to(d)); opt.zero_grad(); l.backward(); opt.step()
        m.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for frames,y,_ in va:
                b,t,c,h,w=frames.shape; emb=m.encode_image(frames.to(d).reshape(b*t,c,h,w))[0].reshape(b,t,-1)
                ps+=torch.sigmoid(m.forward_sequence_embeddings(emb)).cpu().tolist(); ys+=y.tolist()
        th=calibrate_threshold(ys,ps); met=binary_metrics(ys,ps,th); print(f"epoch={e+1} val_auc={met['roc_auc']:.4f}")
        if (met["roc_auc"] or 0)>best:
            best=met["roc_auc"] or 0
            torch.save({"model":m.state_dict(),"threshold":th,"val_metrics":met,"backbone":ck.get("backbone","efficientnet_b4"),"temporal_frames":m.temporal_frames},"models/best_video_model.pth")
    save_metrics({"validation":torch.load("models/best_video_model.pth",map_location="cpu")["val_metrics"]},"models/video_metrics.json")

if __name__=="__main__": main()
