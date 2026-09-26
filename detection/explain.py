from __future__ import annotations
import torch
import torch.nn.functional as F
def gradcam(model,image_tensor):
    activations={}
    def hook(_,__,output): activations["value"]=output; output.retain_grad()
    handle=model.spatial.features[-1].register_forward_hook(hook)
    try:
        x=image_tensor.detach().clone().requires_grad_(True); out=model(x); out["logit"].sum().backward(); fmap=activations["value"]; weights=fmap.grad.mean(dim=(2,3),keepdim=True)
        cam=F.relu((weights*fmap).sum(dim=1,keepdim=True)); cam=F.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False).squeeze(1)
        cam=cam-cam.amin(dim=(1,2),keepdim=True); return (cam/(cam.amax(dim=(1,2),keepdim=True)+1e-6)).detach()
    finally: handle.remove()
