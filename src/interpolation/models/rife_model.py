from __future__ import annotations
from typing import Optional, Dict
import torch 
import torch.nn as nn
from .ifnet import IFNet
from .fusion import FusionNet
from .warp import backward_warp

class SatelliteRIFE(nn.Module):
    def __init__(
            self,
            in_channels:int=1,
            out_channels:int=1,
            base_channels:int=64,
            num_scales:int=3,
            use_external_flow_init:bool=True
    ):
        super().__init__()
        if in_channels!=out_channels:
            raise ValueError(
                "This model preserves channel count end-to-end"
                f"(TIR in == TIR out). Got in={in_channels}, out={out_channels}."
            )
        self.in_channels=in_channels
        self.flow_net=IFNet(
            in_channels=in_channels,
            base_channels=base_channels,
            num_scales=num_scales,
            use_external_flow_init=use_external_flow_init
        )
        self.fusion_net=FusionNet(in_channels=4, c=8)

    def forward(
            self,
            frame0:torch.Tensor,
            frame1:torch.Tensor,
            t:float=0.5,
            flow_01:Optional[torch.Tensor]=None,
            flow_10:Optional[torch.Tensor]=None
    )->Dict[str, torch.tensor]:
        self._validate_input(frame0,frame1)
        init_flow_t0,init_flow_t1=None,None
        if flow_01 is not None and flow_10 is not None:
            init_flow_t0=-t*flow_01
            init_flow_t1=-(1-t)*flow_10
        flow_t0, flow_t1, mask_logit=self.flow_net(
            frame0,
            frame1, init_flow_01=init_flow_t0, init_flow_10=init_flow_t1
        )
        warped0=backward_warp(frame0, flow_t0)
        warped1=backward_warp(frame1, flow_t1)
        mask=torch.sigmoid(mask_logit)
        blended=mask*warped0+(1-mask)*warped1
        residual=self.fusion_net(warped0,warped1,blended,mask)
        pred=torch.clamp(blended+residual,0.0,0.1)
        return {"pred":pred, "flow_t0":flow_t0,"flow_t1":flow_t1,"mask":mask}

    def _validate_input(self, frame0:torch.Tensor, frame1:torch.Tensor)->None:
        if frame0.shape!=frame1.shape:
             raise ValueError(f"frame0/frame1 shape mismatch: {frame0.shape} vs {frame1.shape}")
        if frame0.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {frame0.shape}")
        if frame0.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {frame0.shape[1]}. "
                f"This model is built for single-channel TIR data — do not pass RGB."
            )