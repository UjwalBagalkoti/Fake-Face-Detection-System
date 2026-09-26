from __future__ import annotations
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights,EfficientNet_B4_Weights,efficientnet_b0,efficientnet_b4
class SpatialEncoder(nn.Module):
    def __init__(self,backbone="efficientnet_b4",pretrained=True,out_dim=256):
        super().__init__()
        if backbone=="efficientnet_b4": net=efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT if pretrained else None); feat_dim=1792
        elif backbone=="efficientnet_b0": net=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None); feat_dim=1280
        else: raise ValueError("backbone must be efficientnet_b0 or efficientnet_b4")
        self.features=net.features; self.pool=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Sequential(nn.Flatten(),nn.Linear(feat_dim,out_dim),nn.BatchNorm1d(out_dim),nn.SiLU())
    def forward(self,x):
        fmap=self.features(x); return self.proj(self.pool(fmap)),fmap
class FrequencyEncoder(nn.Module):
    def __init__(self,out_dim=256):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(128,192,3,padding=1),nn.BatchNorm2d(192),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Sequential(nn.Flatten(),nn.Linear(192,out_dim),nn.SiLU())
    @staticmethod
    def to_frequency(x):
        gray=x.mean(dim=1,keepdim=True); mag=torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(gray))))
        mag=(mag-mag.amin(dim=(-2,-1),keepdim=True))/(mag.amax(dim=(-2,-1),keepdim=True)-mag.amin(dim=(-2,-1),keepdim=True)+1e-6)
        return mag.repeat(1,3,1,1)
    def forward(self,x): return self.proj(self.net(self.to_frequency(x)))
class ForensicFusion(nn.Module):
    def __init__(self,backbone="efficientnet_b4",pretrained=True,embedding_dim=256,dropout=.30,temporal_frames=24,temporal_layers=2,temporal_heads=8):
        super().__init__(); self.spatial=SpatialEncoder(backbone,pretrained,embedding_dim); self.frequency=FrequencyEncoder(embedding_dim)
        d=embedding_dim*2; self.image_fuse=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.SiLU(),nn.Dropout(dropout)); self.image_head=nn.Linear(d,1)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=temporal_heads,dim_feedforward=embedding_dim*4,dropout=dropout,batch_first=True,norm_first=True,activation="gelu")
        self.temporal_frames=temporal_frames; self.temporal_pos=nn.Parameter(torch.zeros(1,temporal_frames,d)); nn.init.normal_(self.temporal_pos,std=.02)
        self.temporal=nn.TransformerEncoder(layer,num_layers=temporal_layers); self.temporal_norm=nn.LayerNorm(d); self.temporal_head=nn.Linear(d,1)
    def encode_image(self,x):
        spatial,fmap=self.spatial(x); fused=self.image_fuse(torch.cat([spatial,self.frequency(x)],dim=1)); return fused,fmap
    def forward(self,x):
        fused,fmap=self.encode_image(x); return {"logit":self.image_head(fused).squeeze(1),"embedding":fused,"feature_map":fmap}
    def forward_sequence_embeddings(self,seq):
        n=seq.size(1)
        if n>self.temporal_frames: raise ValueError("sequence exceeds configured temporal_frames")
        z=self.temporal(seq+self.temporal_pos[:,:n]); return self.temporal_head(self.temporal_norm(z.mean(dim=1))).squeeze(1)
