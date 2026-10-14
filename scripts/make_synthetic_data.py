"""Create a tiny synthetic dataset that mirrors the real folder layout.

Used only by the smoke test and unit tests. Primary images follow the Augmentor
naming scheme of Kaggle ``bahribahri/riceleaf``
(``<Class>_original_<source>_<uuid>.<ext>``); external images follow the Paddy
Doctor folder layout. Pixel content is random noise plus a class-specific tint.
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

PRIMARY = {
    "Bacterial leaf blight": ("Bacterial_leaf_blight", "DSC_{:04d}.JPG", (200, 190, 90)),
    "Brown spot": ("Brown_spot", "DSC_{:04d}.jpg", (140, 90, 40)),
    "Health": ("Health", "IMG_20190419_{:06d}.jpg", (60, 160, 60)),
    "Leaf smut": ("Leaf_smut", "DSC_{:04d}.JPG", (70, 70, 70)),
}
EXTERNAL = {"bacterial_leaf_blight": (200, 190, 90), "brown_spot": (140, 90, 40), "normal": (60, 160, 60)}


def _image(rng, tint, size):
    base = np.asarray(tint, dtype=np.float32)[None, None, :]
    noise = rng.normal(0, 35, (size, size, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def make(root, sources_per_class=8, copies=5, external_per_class=6, size=64, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for cls, (prefix, src_fmt, tint) in PRIMARY.items():
        d = root / "riceleaf" / cls
        d.mkdir(parents=True, exist_ok=True)
        for s in range(sources_per_class):
            src = src_fmt.format(100 + s)
            for _ in range(copies):
                uid = uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63)))
                ext = src.rsplit(".", 1)[1]
                _image(rng, tint, size).save(d / f"{prefix}_original_{src}_{uid}.{ext}")
    for folder, tint in EXTERNAL.items():
        d = root / "paddy" / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(external_per_class):
            _image(rng, tint, size * 2).save(d / f"{100000 + i}.jpg")
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke_data")
    ap.add_argument("--sources", type=int, default=8)
    ap.add_argument("--copies", type=int, default=5)
    a = ap.parse_args()
    print(f"synthetic data written to {make(a.out, a.sources, a.copies)}")
