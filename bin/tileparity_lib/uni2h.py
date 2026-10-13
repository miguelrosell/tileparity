"""Processes 1a + 2a - UNI2-h adapter and embedding.

Adapter (1a), applied to each tile exactly as read from the resampled image:

1. the tile is cut at (x0, y0) with side 224 px - no resizing, no cropping;
2. pixel values are scaled to [0, 1] (divide by 255);
3. each channel is normalised as (x - mean) / std with the ImageNet statistics
   mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), as in the UNI2-h model card.

Embedding (2a): UNI2-h (ViT-H/14, 24 blocks, 1536-d, 8 register tokens) cuts the
224 x 224 input into 16 x 16 = 256 patches of 14 x 14 px internally and returns the
class token: one 1536-dimensional vector per tile.

The transform is written out explicitly instead of using timm's ``create_transform``
so that no resize or centre crop can change the field of view.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 1536
HF_REPO = "MahmoodLab/UNI2-h"


def timm_kwargs(test_mode: bool = False) -> dict:
    import timm
    import torch

    kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": EMBED_DIM,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    if test_mode:
        kwargs["depth"] = 1  # same input/output contract, random weights, fast on CPU
    return kwargs


def build_model(weights: str | None, test_mode: bool = False):
    """UNI2-h from a local weights file, from the Hugging Face hub, or a 1-block random
    stand-in (``test_mode``) for pipeline tests without access to the gated weights."""
    import timm
    import torch

    if test_mode:
        torch.manual_seed(0)
        return timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs(True)), "test-mode-random-1-block"
    if weights:
        model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs())
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model, f"file:{Path(weights).name}"
    model = timm.create_model(f"hf-hub:{HF_REPO}", pretrained=True, **timm_kwargs())
    return model, f"hf-hub:{HF_REPO}"


def sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class TileDataset:
    """Reads RGB tiles from the resampled image; the libvips handle is opened lazily in
    each data-loader worker."""

    def __init__(self, image_path: str, tiles: pd.DataFrame, tile_px: int):
        self.image_path = image_path
        self.x0 = tiles["x0_px"].to_numpy(int)
        self.y0 = tiles["y0_px"].to_numpy(int)
        self.tile_px = tile_px
        self._img = None

    def __len__(self) -> int:
        return len(self.x0)

    def __getitem__(self, i: int):
        import torch

        if self._img is None:
            import pyvips

            os.environ.setdefault("VIPS_CONCURRENCY", "1")
            self._img = pyvips.Image.new_from_file(self.image_path, access="random")
        from .image import read_region

        rgb = read_region(self._img, int(self.x0[i]), int(self.y0[i]), self.tile_px, self.tile_px)[:, :, :3]
        if rgb.shape != (224, 224, 3):
            raise ValueError(f"UNI2-h needs 224x224x3 tiles, got {rgb.shape}")
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std


def embed(
    image_path: str,
    tiles: pd.DataFrame,
    model,
    device: str = "auto",
    precision: str = "fp16",
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[np.ndarray, dict]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is visible (enable the 'gpu' profile?)")
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    if device == "cpu" and dtype is torch.float16:
        dtype = None  # half precision on CPU is slow and poorly supported
    model = model.to(device).eval()
    dataset = TileDataset(image_path, tiles, 224)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=device == "cuda",
    )
    out = np.zeros((len(dataset), EMBED_DIM), dtype=np.float32)
    start = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if dtype is not None:
                with torch.autocast(device_type=device, dtype=dtype):
                    emb = model(batch)
            else:
                emb = model(batch)
            emb = emb.float().cpu().numpy()
            out[start : start + len(emb)] = emb
            start += len(emb)
    info = {"device": device, "precision": precision if dtype is not None else "fp32",
            "torch": torch.__version__, "batch_size": batch_size}
    return out, info
