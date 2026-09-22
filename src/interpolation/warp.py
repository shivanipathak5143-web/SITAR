from __future__ import annotations 
import torch
import torch.nn.functional as F

def backward_warp(image:torch.Tensor, flow:torch.Tensor)->torch.Tensor:
     if image.dim() != 4 or flow.dim() != 4:
        raise ValueError(
            f"Expected 4D tensors (B,C,H,W). Got image={image.shape}, flow={flow.shape}"
        )
     if flow.shape[1] != 2:
        raise ValueError(f"flow must have 2 channels (dx, dy), got shape {flow.shape}")
     if image.shape[0] != flow.shape[0] or image.shape[2:] != flow.shape[2:]:
        raise ValueError(
            f"image and flow spatial/batch dims must match. "
            f"image={image.shape}, flow={flow.shape}"
        )

     B, C, H, W = image.shape
     device = image.device

     y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
     grid = torch.stack((x_coords, y_coords), dim=0)          # (2, H, W)
     grid = grid.unsqueeze(0).expand(B, -1, -1, -1)            # (B, 2, H, W)

     new_grid = grid + flow

    # Normalize to [-1, 1] for grid_sample
     new_grid_x = 2.0 * new_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
     new_grid_y = 2.0 * new_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
     sample_grid = torch.stack((new_grid_x, new_grid_y), dim=-1)  # (B, H, W, 2)

     warped = F.grid_sample(
        image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
    )
     return warped    
