"""
Training entry point for the satellite frame interpolation model.

Usage:
    python -m src.interpolation.train --config configs/interpolation.yaml
    python -m src.interpolation.train --config configs/interpolation.yaml --resume checkpoints/interpolation_latest.pth
"""

from __future__ import annotations
import os
import argparse
import random
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SatelliteFrameDataset
from .models.rife_model import SatelliteRIFE
from .losses import CombinedInterpolationLoss
from .utils import load_config, NormalizationConfig, compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests device='cuda' but CUDA is not available on this machine. "
            "Set device: cpu or device: auto in the config."
        )
    return torch.device(device_cfg)


def build_dataloaders(cfg: Dict[str, Any]):
    norm_cfg = NormalizationConfig.from_dict(cfg["data"]["normalization"])

    common_kwargs = dict(
        data_dir=cfg["data"]["train_dir"],
        image_variable=cfg["data"]["image_variable"],
        image_size=cfg["data"]["image_size"],
        crop_size=cfg["data"]["crop_size"],
        stride=cfg["data"]["stride"],
        val_fraction=cfg["data"]["val_fraction"],
        normalization_cfg=norm_cfg,
    )

    train_ds = SatelliteFrameDataset(split="train", augment=True, **common_kwargs)
    val_ds = SatelliteFrameDataset(split="val", augment=False, **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return train_loader, val_loader


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    m = cfg["model"]
    return SatelliteRIFE(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        base_channels=m["base_channels"],
        num_scales=m["num_scales"],
        use_external_flow_init=m["use_external_flow_init"],
    )


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    t = cfg["training"]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    elif t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {t['optimizer']}")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    t = cfg["training"]
    if t["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    elif t["scheduler"] == "none" or t["scheduler"] is None:
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {t['scheduler']}")


def load_external_flow(path0: str, path1: str):
    """
    Hook for Member 1's RAFT optical flow module.

    Currently returns (None, None), meaning the model runs WITHOUT external
    flow initialization by default. To integrate RAFT once Member 1's module
    is ready, replace this function body with a call into
    `src.optical_flow`, e.g.:

        from src.optical_flow.raft_infer import estimate_flow
        flow_01, flow_10 = estimate_flow(path0, path1)
        return flow_01, flow_10

    Keep the return type as (torch.Tensor or None, torch.Tensor or None),
    shape (B, 2, H, W) each, matching the batch's frame0/frame1 resolution.
    """
    return None, None


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_metrics: Dict[str, float],
    cfg: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu") -> int:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def validate(model, val_loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metric_sums = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            frame0 = batch["frame0"].to(device)
            frame1 = batch["frame1"].to(device)
            target = batch["target"].to(device)

            out = model(frame0, frame1, t=0.5)
            pred = out["pred"]

            losses = loss_fn(pred, target)
            total_loss += losses["total"].item()

            pred_np = pred.squeeze(1).cpu().numpy()
            target_np = target.squeeze(1).cpu().numpy()
            for b in range(pred_np.shape[0]):
                m = compute_metrics(pred_np[b], target_np[b], data_range=1.0)
                for k in metric_sums:
                    metric_sums[k] += m[k]
            n_batches += 1

    n_samples = n_batches * val_loader.batch_size
    avg_metrics = {k: v / max(n_samples, 1) for k, v in metric_sums.items()}
    avg_metrics["val_loss"] = total_loss / max(n_batches, 1)
    return avg_metrics


def train(cfg: Dict[str, Any], resume_path: Optional[str] = None) -> None:
    set_seed(cfg["training"]["seed"])
    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = CombinedInterpolationLoss(
        w_recon=cfg["loss"]["reconstruction_weight"],
        w_ssim=cfg["loss"]["ssim_weight"],
        w_grad=cfg["loss"]["gradient_weight"],
    )

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_metric_value = -float("inf")
    epochs_without_improvement = 0

    if resume_path is not None:
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, device=device) + 1
        print(f"Resumed from {resume_path}, continuing at epoch {start_epoch}")

    ckpt_dir = cfg["checkpoint"]["directory"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_metric_name = cfg["training"]["best_metric"]

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            frame0 = batch["frame0"].to(device)
            frame1 = batch["frame1"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(frame0, frame1, t=0.5)
                losses = loss_fn(out["pred"], target)
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)

        if (epoch + 1) % cfg["training"]["validate_every"] == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            print(
                f"Epoch {epoch + 1}/{cfg['training']['epochs']} "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_metrics['val_loss']:.4f} "
                f"Val MSE: {val_metrics['mse']:.5f} "
                f"Val PSNR: {val_metrics['psnr']:.2f} "
                f"Val SSIM: {val_metrics['ssim']:.4f}"
            )

            current_value = val_metrics.get(best_metric_name, val_metrics["val_loss"])
            improved = current_value > best_metric_value

            if improved:
                best_metric_value = current_value
                epochs_without_improvement = 0
                best_path = os.path.join(ckpt_dir, cfg["checkpoint"]["best_filename"])
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_metrics, cfg)
                print(f"  New best ({best_metric_name}={current_value:.4f}) -> saved {best_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping: no improvement in {best_metric_name} for "
                      f"{cfg['training']['early_stopping_patience']} validation rounds.")
                break
        else:
            print(f"Epoch {epoch + 1}/{cfg['training']['epochs']} Train Loss: {train_loss:.4f}")

        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
            latest_path = os.path.join(ckpt_dir, cfg["checkpoint"]["latest_filename"])
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch, {}, cfg)


def main():
    parser = argparse.ArgumentParser(description="Train the satellite frame interpolation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Optional path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()