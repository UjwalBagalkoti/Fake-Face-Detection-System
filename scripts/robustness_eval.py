from __future__ import annotations
import argparse,io,json
from pathlib import Path
import torch
from PIL import Image,ImageFilter
from torch.utils.data import DataLoader,Dataset
from detection.model import ForensicFusion
from training.datasets import eval_transform,build_imagefolder
from training.metrics import binary_metrics

def jpeg_roundtrip(image,quality):
    buf=io.BytesIO(); image.save(buf,format="JPEG",quality=quality); buf.seek(0); return Image.open(buf).convert("RGB")

class VariantDataset(Dataset):
    def __init__(self,samples,transform):
        self.samples=samples; self.transform=transform
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        path,label=self.samples[index]
        return self.transform(Image.open(path).convert("RGB")),label

def evaluate(model,dataset,device,threshold):
    loader=DataLoader(dataset,batch_size=32,shuffle=False)
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            ps.extend(torch.sigmoid(model(x.to(device))["logit"]).cpu().tolist())
            ys.extend(y.tolist())
    return binary_metrics(ys,ps,threshold)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--split",default="test")
    a=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.model,map_location=device)
    model=ForensicFusion(backbone=ck.get("backbone","efficientnet_b4"),pretrained=False).to(device)
    model.load_state_dict(ck.get("model",ck),strict=False); model.eval()
    base=build_imagefolder(Path(a.data)/a.split,False)
    variants={
        "native":lambda im:im,
        "jpeg70":lambda im:jpeg_roundtrip(im,70),
        "jpeg40":lambda im:jpeg_roundtrip(im,40),
        "blur_1px":lambda im:im.filter(ImageFilter.GaussianBlur(1.0))
    }
    results={}
    for name,fn in variants.items():
        results[name]=evaluate(model,VariantDataset(base.samples,lambda im,fn=fn:eval_transform()(fn(im))),device,float(ck.get("threshold",.5)))
    Path("models").mkdir(exist_ok=True)
    Path("models/robustness_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
