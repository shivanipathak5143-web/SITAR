from __future__ import annotations
from typing import Optional, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F

def _conv(in_ch:int, out_ch:int, k:int=3, s:int=1, p:int=1)->nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k,s,p),
        nn.PReLU(out_ch)
    )

class IFBlock(nn.Module):
    def __init(self,in_channels:int, c:int=64):
        super().__init__()
        self.conv0=nn.Sequential(
            _conv(in_channels, c// 2,3,2,1),
            _conv(c //2, c, 3,2,1),
        )
        self.resblocks=nn.Sequential(
            _conv(c,c,3,1,1),
            _conv(c,c,3,1,1),
            _conv(c,c,3,1,1),
            _conv(c,c,3,1,1),
        )
        self.up1=nn.ConvTranspose(c,c // 2,4,2,1)
        self.act1=nn.PReLU(c // 2)
        self.up2=nn.ConvTranspose(c //2,5,4,2,1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        feat=self.conv0(x)
        feat=self.resblocks(feat)+feat
        feat=self.act1(self.up1(feat))
        out=self.up2(feat)
        return out

class IFNet(nn.Module):
    def __init__(
            self,
            in_channels:int=1,
            base_channels:int=64,
            num_scales:int=3,
            use_external_flow_init:bool=True
    ):
        super().__init__()
        self.in_channels=in_channels
        self.num_scales=num_scales
        self.use_external_flow_init=use_external_flow_init
        frames_channels=in_channels*2
        blocks=[]
        for i in range(num_scales):
            if i==0:
                block_in=frames_channels
            else:
                block_in=frames_channels+4
            blocks.append(IFBlock(block_in, c=base_channels))
            self.blocks=nn.ModuleList(blocks)
            self.scale_factors=[2**(num_scales-1-i) for i in range(num_scales)]

    def forward(
            self,
            frame0:torch.Tensor,
            frame1:torch.Tensor,
            init_flow_01:Optional[torch.Tensor]=None,
            init_flow_10: Optional[torch.Tensor]=None
    )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B,C,H,W=frame0.shape
        x_full=torch.cat([frame0,frame1],dim=1)
        flow=None
        for i, (block,sf) in enumerate(zip(self.blocks,self.scale_factors)):
            h_s,w_s=max(H // sf,1), max(W // sf,1)
            x_s=F.interpolate(flow,size=(h_s,w_s), mode='bilinear', align_corners=False)

            if i==0:
                block_input=x_s
            else:
                flow_s=F.interpolate(flow, size=(h_s,w_s), mode='bilinear',align_corners=False)
                scale_ratio=w_s/flow.shape[-1]
                flow_s=flow_s*scale_ratio
                block_input=torch.cat([x_s,flow_s],dim=1)
            out=block(block_input)
            out=F.interpolate(out, size=(h_s,w_s), mode='bilinear', align_corners=False)

            new_flow=out[:,:4]
            mask_logit_s=out[:, 4:5]
            if i==0:
                flow=new_flow
                if self.use_external_flow_init and init_flow_01 is not None and init_flow_10 is not None:
                    ext_flow=torch.cat([init_flow_01,init_flow_10],dim=1)
                    ext_flow_s=F.interpolate(ext_flow,size=(h_s,w_s), mode='bilinear', align_corners=False)
                    ext_flow_s=ext_flow_s*(w_s/ init_flow_01.shape[-1])
                    flow=flow+ext_flow_s
                else:
                    flow_prev_resized=F.interpolate(flow,size=(h_s,w_s), mode='bilinear',align_corners=False)
                    flow_prev_resized=flow_prev_resized*(w_s/flow.shape[-1])
                    flow=flow_prev_resized+new_flow

                mask_logit=mask_logit_s

            flow_full=F.interpolate(flow, size=(H,W), mode='bilinear',align_corners=False)
            flow_full=flow_full*(W/flow.shape[-1])
            mask_full=F.interpolate(mask_logit, size=(H,W), mode='bilinear', align_corners=False)

            flow_t0 = flow_full[:, 0:2]
        flow_t1 = flow_full[:, 2:4]
        return flow_t0, flow_t1, mask_full

    
