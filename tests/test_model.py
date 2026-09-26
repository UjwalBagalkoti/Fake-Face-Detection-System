import torch
from detection.model import ForensicFusion
def test_b0_forward():
    m=ForensicFusion(backbone="efficientnet_b0",pretrained=False,embedding_dim=64,temporal_frames=4,temporal_heads=4); m.eval(); x=torch.randn(1,3,224,224); out=m(x)
    assert out["logit"].shape==(1,) and out["embedding"].shape==(1,128); assert m.forward_sequence_embeddings(out["embedding"].unsqueeze(1).repeat(1,4,1)).shape==(1,)
