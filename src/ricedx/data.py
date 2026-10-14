"""Dataset indexing, source-photo grouping and leakage-free splitting.

The primary dataset (Kaggle ``bahribahri/riceleaf``) contains 16,000 images that
were produced by the Augmentor library from 160 original photographs. Augmentor
names each output ``<Class>_original_<source file>_<uuid>.<ext>``, so the source
photograph of every image can be recovered from its filename. Any split that
puts augmented copies of the same source photograph in both the training and the
test partition leaks information; the group-aware splits below prevent that.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
AUGMENTOR_RE = re.compile(
    r"^(?P<prefix>.+?)_original_(?P<source>.+?)_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.[A-Za-z]+$"
)


# --------------------------------------------------------------------------- #
# Locating datasets
# --------------------------------------------------------------------------- #
def _contains_all(folder: Path, names) -> bool:
    return all((folder / n).is_dir() for n in names)


def find_dataset_root(setting, required_subdirs, search_roots=("/kaggle/input", "data")):
    """Return the folder containing every name in ``required_subdirs``.

    ``setting`` is a path, ``None`` (dataset disabled) or ``"auto"`` (search
    ``search_roots`` recursively, at most 5 levels deep).
    """
    if setting is None:
        return None
    if setting != "auto":
        p = Path(setting)
        if not _contains_all(p, required_subdirs):
            raise FileNotFoundError(f"{p} does not contain {list(required_subdirs)}")
        return p
    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > 5:
                dirnames[:] = []
                continue
            if _contains_all(Path(dirpath), required_subdirs):
                return Path(dirpath)
    return None


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def source_of(filename: str) -> str | None:
    """Source photograph encoded in an Augmentor filename, or None."""
    m = AUGMENTOR_RE.match(filename)
    return m.group("source") if m else None


def phash(path, hash_size: int = 8, highfreq: int = 4) -> int:
    """64-bit DCT perceptual hash (used only when filenames carry no source)."""
    from scipy.fft import dct

    size = hash_size * highfreq
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float64)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _phash_groups(paths, max_distance: int = 6):
    """Union-find over perceptual hashes: near-duplicates share a group id."""
    hashes = [phash(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= max_distance:
                parent[find(i)] = find(j)
    return [f"phash{find(i)}" for i in range(len(paths))]


def build_index(root, classes) -> pd.DataFrame:
    """One row per image: path, label (int), class_name, source, group."""
    rows = []
    root = Path(root)
    for label, cls in enumerate(classes):
        files = sorted(
            f for f in (root / cls).iterdir() if f.suffix.lower() in IMG_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No images in {root / cls}")
        sources = [source_of(f.name) for f in files]
        if any(s is None for s in sources):
            sources = _phash_groups(files)
        for f, s in zip(files, sources):
            rows.append(
                {
                    "path": str(f),
                    "label": label,
                    "class_name": cls,
                    "source": s,
                    "group": f"{label}:{s}",
                }
            )
    return pd.DataFrame(rows)


def build_external_index(root, external_map, classes, max_per_class=None, seed=0):
    """Index of an external dataset whose folders map onto primary classes."""
    rows = []
    root = Path(root)
    rng = np.random.default_rng(seed)
    for folder, target in external_map.items():
        files = sorted(
            f for f in (root / folder).iterdir() if f.suffix.lower() in IMG_EXTS
        )
        if max_per_class is not None and len(files) > max_per_class:
            files = [files[i] for i in sorted(rng.choice(len(files), max_per_class, replace=False))]
        label = classes.index(target)
        for f in files:
            rows.append(
                {
                    "path": str(f),
                    "label": label,
                    "class_name": target,
                    "source": f.name,
                    "group": f"ext:{folder}:{f.name}",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def random_split(df, test_fraction, seed):
    """Image-level stratified split (the thesis protocol; leaks across groups)."""
    tr, te = train_test_split(
        np.arange(len(df)), test_size=test_fraction, stratify=df["label"], random_state=seed
    )
    return np.sort(tr), np.sort(te)


def group_split(df, test_fraction, seed):
    """Source-grouped stratified split: no source photograph spans both parts."""
    n_splits = max(2, int(round(1.0 / test_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(df)), df["label"], df["group"]))
    return np.sort(tr), np.sort(te)


def group_kfold(df, n_folds, seed):
    """Stratified group k-fold; every source photograph is tested exactly once."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (np.sort(tr), np.sort(te))
        for tr, te in sgkf.split(np.zeros(len(df)), df["label"], df["group"])
    ]


def split_train_val(df, idx, val_fraction, seed, grouped=True):
    """Split ``idx`` into train/val, grouped or image-level."""
    sub = df.iloc[idx].reset_index(drop=True)
    if grouped:
        tr, va = group_split(sub, val_fraction, seed)
    else:
        tr, va = random_split(sub, val_fraction, seed)
    return idx[tr], idx[va]


def assert_disjoint_groups(df, a_idx, b_idx):
    shared = set(df.iloc[a_idx]["group"]) & set(df.iloc[b_idx]["group"])
    if shared:
        raise AssertionError(f"{len(shared)} source groups appear in both partitions")


def group_overlap_fraction(df, train_idx, test_idx) -> float:
    """Fraction of test images whose source photograph also appears in training."""
    train_groups = set(df.iloc[train_idx]["group"])
    return float(df.iloc[test_idx]["group"].isin(train_groups).mean())


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
def load_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class ImageDataset:
    """Minimal map-style dataset; optional in-RAM cache of decoded images."""

    def __init__(self, df, transform, cache=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = transform
        self.cache = cache  # dict path -> PIL image, shared across splits

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = self.cache[p] if self.cache is not None and p in self.cache else load_rgb(p)
        return self.transform(img), int(self.labels[i])


def build_cache(df) -> dict:
    return {p: load_rgb(p) for p in df["path"]}
