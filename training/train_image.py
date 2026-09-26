from __future__ import annotations
import argparse,random
from pathlib import Path
import numpy as np,torch
from torch import nn,optim
from torch.utils.data import DataLoader
from detection.model import ForensicFusion
from training.datasets import build_imagefolder
from training.metrics import binary_metrics,calibrate_threshold,save_metrics
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def epoch(model,loader,opt,loss_fn,device,train):
    model.train(train); ys=[]; ps=[]; total=0
    for x,y in loader:
        x=x.to(device); y=y.float().to(device)
        if train: opt.zero_grad(set_to_none=True)
        logit=model(x)["logit"]; loss=loss_fn(logit,y)
        if train: loss.backward(); opt.step()
        total+=loss.item()*x.size(0); ys+=y.detach().cpu().tolist(); ps+=torch.sigmoid(logit).detach().cpu().tolist()
    return total/len(loader.dataset),ys,ps
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--backbone",default="efficientnet_b4",choices=["efficientnet_b0","efficientnet_b4"]); ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--batch-size",type=int,default=24); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--pretrained",action="store_true"); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    seed_everything(a.seed); data=Path(a.data); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=build_imagefolder(data/"train",True); va=build_imagefolder(data/"val"); te=build_imagefolder(data/"test")
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.workers); vl=DataLoader(va,batch_size=a.batch_size,num_workers=a.workers); el=DataLoader(te,batch_size=a.batch_size,num_workers=a.workers)
    m=ForensicFusion(backbone=a.backbone,pretrained=a.pretrained).to(device); labels=np.array([y for _,y in tr.samples]); pos=max(1,int((labels==1).sum())); neg=max(1,int((labels==0).sum()))
    loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device)); opt=optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-5); best=-1; Path("models").mkdir(exist_ok=True)
    for e in range(1,a.epochs+1):
        trl,_,_=epoch(m,tl,opt,loss,device,True); vl_loss,vy,vp=epoch(m,vl,opt,loss,device,False); th=calibrate_threshold(vy,vp); met=binary_metrics(vy,vp,th); print(f"epoch={e} train_loss={trl:.4f} val_auc={met['roc_auc']:.4f}")
        if (met["roc_auc"] or 0)>best:
            best=met["roc_auc"] or 0; torch.save({"model":m.state_dict(),"threshold":th,"backbone":a.backbone,"val_metrics":met}, "models/best_image_model.pth")
    ck=torch.load("models/best_image_model.pth",map_location=device); m.load_state_dict(ck["model"]); _,ty,tp=epoch(m,el,opt,loss,device,False); save_metrics({"validation":ck["val_metrics"],"test":binary_metrics(ty,tp,ck["threshold"])}, "models/image_metrics.json")
if __name__=="__main__": main()
