from __future__ import annotations
import torch 
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self,in_channels:int=4, c:int=6):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,c,3,1,1),nn.PReLU(c),
            nn.Conv2d(c,c,3,1,1),nn.PReLU(c),
            nn.Conv2d(c,c,3,1,1),nn.PReLU(c),
            nn.Conv2d(c,c//2,3,1,1),nn.PReLU(c//2),
            nn.Conv2d(c//2,1,3,1,1),
        )

    def forward(self,warped0,warped1,blended,mask):
        x=torch.cat([warped0,warped1,blended,mask],dim=1)
        residual=self.net(x)
        return residual
