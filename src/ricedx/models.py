"""Model factory and preprocessing transforms (timm backbones)."""
from __future__ import annotations

import timm
import torch
from torchvision import transforms as T

VIT_PREFIXES = ("deit", "vit")


def create_model(model_cfg, num_classes, pretrained, img_size):
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    if model_cfg["timm"].startswith(VIT_PREFIXES):
        kwargs["img_size"] = img_size
    return timm.create_model(model_cfg["timm"], **kwargs)


def norm_stats(model):
    cfg = timm.data.resolve_model_data_config(model)
    return tuple(cfg["mean"]), tuple(cfg["std"])


def train_transform(img_size, mean, std):
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def eval_transform(img_size, mean, std):
    return T.Compose(
        [
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def resolve_img_size(model_cfg, train_cfg):
    return train_cfg.get("img_size_override") or model_cfg["img_size"]


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model, img_size) -> int:
    """Forward FLOPs for one image (FlopCounterMode counts a multiply-add as 2)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=next(model.parameters()).device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return int(fc.get_total_flops())
