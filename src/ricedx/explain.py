"""Grad-CAM on the last convolutional feature map of a timm CNN."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_rgb


def gradcam(model, x, target=None):
    """Return (heatmap HxW in [0,1], predicted class, probability)."""
    model.eval()
    feats = model.forward_features(x)
    if feats.ndim != 4:
        raise ValueError("Grad-CAM needs a 4-D convolutional feature map")
    feats.retain_grad()
    logits = model.forward_head(feats)
    probs = logits.softmax(1)
    cls = int(logits.argmax(1)) if target is None else int(target)
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), cls, float(probs[0, cls].detach())


def gradcam_figure(model, rows, transform, img_size, class_names, out_path, device):
    """Grid of image / Grad-CAM overlays for the given dataframe rows."""
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.0), squeeze=False)
    for j, (_, r) in enumerate(rows.iterrows()):
        img = load_rgb(r["path"])
        x = transform(img).unsqueeze(0).to(device)
        cam, cls, p = gradcam(model, x)
        shown = np.asarray(img.resize((img_size, img_size)))
        axes[0, j].imshow(shown)
        axes[0, j].set_title(f"true: {class_names[r['label']]}", fontsize=8)
        axes[1, j].imshow(shown)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"pred: {class_names[cls]} ({p:.2f})", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
