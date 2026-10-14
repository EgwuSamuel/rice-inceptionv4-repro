"""Training and inference loops."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import ImageDataset
from .models import create_model, eval_transform, norm_stats, resolve_img_size, train_transform


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(ds, batch_size, shuffle, num_workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_logits(model, df, transform, cfg, cache=None, device=None):
    device = device or get_device()
    model.eval()
    ds = ImageDataset(df, transform, cache)
    dl = _loader(ds, cfg["train"]["batch_size"] * 2, False, cfg["data"]["num_workers"], 0)
    out = []
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    for x, _ in dl:
        with torch.autocast(device.type, enabled=use_amp):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def fit(model_cfg, cfg, train_df, val_df, seed, cache=None, log=print):
    """Train one model; return (best model, history list, info dict)."""
    seed_everything(seed)
    device = get_device()
    tc = cfg["train"]
    n_classes = len(cfg["data"]["classes"])
    img_size = resolve_img_size(model_cfg, tc)
    model = create_model(model_cfg, n_classes, tc["pretrained"], img_size).to(device)
    mean, std = norm_stats(model)
    t_train = train_transform(img_size, mean, std)
    t_eval = eval_transform(img_size, mean, std)

    dl = _loader(
        ImageDataset(train_df, t_train, cache),
        tc["batch_size"], True, cfg["data"]["num_workers"], seed,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, tc["epochs"] * len(dl)))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        tot_loss, tot_n, tot_correct = 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(y)
            tot_n += len(y)
            tot_correct += (logits.argmax(1) == y).sum().item()

        val_logits = predict_logits(model, val_df, t_eval, cfg, cache, device)
        val_pred = val_logits.argmax(1)
        val_y = val_df["label"].to_numpy()
        val_loss = float(
            torch.nn.functional.cross_entropy(torch.tensor(val_logits), torch.tensor(val_y)).item()
        )
        val_f1 = float(f1_score(val_y, val_pred, average="macro", zero_division=0))
        rec = {
            "epoch": epoch,
            "train_loss": tot_loss / tot_n,
            "train_acc": tot_correct / tot_n,
            "val_loss": val_loss,
            "val_acc": float((val_pred == val_y).mean()),
            "val_macro_f1": val_f1,
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)
        log(
            f"  ep {epoch:02d} loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
            f"val loss {val_loss:.4f} acc {rec['val_acc']:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= tc["patience"]:
                log(f"  early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(best_state)
    info = {
        "img_size": img_size,
        "mean": list(mean),
        "std": list(std),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_time_s": time.time() - t0,
        "device": str(device),
    }
    return model, history, info, t_eval
