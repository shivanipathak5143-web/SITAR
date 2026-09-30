"""
PyTorch Dataset for building (frame0, target, frame1) training triplets from
a directory of temporally-ordered satellite frames.

Design rules enforced here (per project requirements):
- Frames are NEVER randomly paired; triplets are built strictly from
  consecutive frames in chronological order.
- Train/val split is chronological (earlier frames = train, later = val),
  NOT a random shuffle, to avoid leaking near-duplicate adjacent frames
  across the split.
- The same geometric augmentation (flip/crop) is applied identically to all
  three frames in a triplet, so spatial correspondence is preserved.
"""

from __future__ import annotations
import os
import re
import glob
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_satellite_frame, normalize_frame, NormalizationConfig


_TIMESTAMP_PATTERN = re.compile(r"(\d{8})[_\-]?(\d{4,6})")  # e.g. 20260101_0010 or 20260101001000


def _extract_sort_key(path: str) -> str:
    """
    Extract a sortable timestamp string from the filename if possible.
    Falls back to the filename itself (lexical sort) if no timestamp pattern
    is found — the caller is responsible for ensuring filenames sort
    chronologically in that case.
    """
    fname = os.path.basename(path)
    match = _TIMESTAMP_PATTERN.search(fname)
    if match:
        return match.group(1) + match.group(2)
    return fname


class SatelliteFrameDataset(Dataset):
    """
    Args:
        data_dir: directory containing .nc/.h5 frame files, evenly spaced in time
        image_variable: variable name to read (None = auto-detect)
        image_size: if set, frames are resized to (image_size, image_size)
        crop_size: if set, a random (train) or center (val) crop is taken after resize
        stride: step between consecutive triplets (1 = maximum overlap)
        split: "train" or "val"
        val_fraction: fraction of the CHRONOLOGICAL sequence reserved for validation
                      (taken from the END of the sequence, not randomly sampled)
        augment: whether to apply flips (only meaningful for split="train")
        normalization_cfg: NormalizationConfig instance
    """

    def __init__(
        self,
        data_dir: str,
        image_variable: Optional[str] = None,
        image_size: Optional[int] = 256,
        crop_size: Optional[int] = None,
        stride: int = 1,
        split: str = "train",
        val_fraction: float = 0.15,
        augment: bool = True,
        normalization_cfg: Optional[NormalizationConfig] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

        self.image_variable = image_variable
        self.image_size = image_size
        self.crop_size = crop_size
        self.stride = max(1, stride)
        self.split = split
        self.augment = augment and split == "train"
        self.norm_cfg = normalization_cfg or NormalizationConfig()

        files = sorted(
            glob.glob(os.path.join(data_dir, "*.nc")) + glob.glob(os.path.join(data_dir, "*.h5")),
            key=_extract_sort_key,
        )
        if len(files) < 3:
            raise ValueError(
                f"Need at least 3 chronologically ordered frames to build a "
                f"single triplet. Found {len(files)} in {data_dir}."
            )

        split_idx = int(len(files) * (1 - val_fraction))
        split_idx = max(split_idx, 2)  # ensure at least 2 frames remain for whichever split needs them

        if split == "train":
            self.files = files[:split_idx]
        else:
            self.files = files[split_idx:]

        if len(self.files) < 3:
            raise ValueError(
                f"After chronological {split} split, only {len(self.files)} frames "
                f"remain — need at least 3. Adjust val_fraction or provide more data."
            )

        self.triplet_indices: List[Tuple[int, int, int]] = [
            (i, i + 1, i + 2) for i in range(0, len(self.files) - 2, self.stride)
        ]

    def __len__(self) -> int:
        return len(self.triplet_indices)

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        data, _, _ = load_satellite_frame(path, var_name=self.image_variable)
        normed, _ = normalize_frame(data, self.norm_cfg)
        return normed

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        if self.image_size is None:
            return arr
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return t.squeeze(0).squeeze(0).numpy()

    def _apply_joint_crop_and_flip(
        self, arrs: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f0, ft, f1 = arrs
        H, W = f0.shape

        if self.crop_size is not None and self.crop_size < H and self.crop_size < W:
            if self.split == "train":
                top = np.random.randint(0, H - self.crop_size + 1)
                left = np.random.randint(0, W - self.crop_size + 1)
            else:
                top = (H - self.crop_size) // 2
                left = (W - self.crop_size) // 2
            sl = (slice(top, top + self.crop_size), slice(left, left + self.crop_size))
            f0, ft, f1 = f0[sl], ft[sl], f1[sl]

        if self.augment:
            # Horizontal flip: safe — mirrors east-west, preserves physical meaning.
            if np.random.rand() < 0.5:
                f0, ft, f1 = f0[:, ::-1].copy(), ft[:, ::-1].copy(), f1[:, ::-1].copy()
            # Vertical flip: ONLY safe if your projection doesn't encode a fixed
            # north-up orientation assumption elsewhere in the pipeline (e.g. the
            # dashboard/visualization). Enabled by default here since flipping
            # doesn't change *relative* cloud-motion learning, but disable this
            # block if absolute geographic orientation must be preserved for
            # downstream georeferencing.
            if np.random.rand() < 0.5:
                f0, ft, f1 = f0[::-1, :].copy(), ft[::-1, :].copy(), f1[::-1, :].copy()
            # NOTE: We deliberately do NOT apply rotation or color jitter.
            # Rotation would require re-deriving lat/lon geolocation (not
            # attempted here) and color jitter is meaningless/harmful for
            # single-channel physical temperature data.

        return f0, ft, f1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i0, it, i1 = self.triplet_indices[idx]
        path0, path_target, path1 = self.files[i0], self.files[it], self.files[i1]

        f0 = self._resize(self._load_and_preprocess(path0))
        ft = self._resize(self._load_and_preprocess(path_target))
        f1 = self._resize(self._load_and_preprocess(path1))

        f0, ft, f1 = self._apply_joint_crop_and_flip((f0, ft, f1))

        frame0 = torch.from_numpy(f0).unsqueeze(0).float()   # (1, H, W)
        target = torch.from_numpy(ft).unsqueeze(0).float()
        frame1 = torch.from_numpy(f1).unsqueeze(0).float()

        return {
            "frame0": frame0,
            "frame1": frame1,
            "target": target,
            "t": 0.5,  # equal-spacing assumption; see README note if spacing varies
            "path0": path0,
            "path_target": path_target,
            "path1": path1,
        }