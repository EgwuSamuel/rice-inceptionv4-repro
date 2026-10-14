"""Run every experiment of the study and write tables, figures and a summary.

Stages (run in this order; each one is resumable and skips finished runs):
  leakage    E1  image-level random split vs source-grouped split (InceptionV4)
  benchmark  E2  stratified group k-fold CV over source photographs, all models
                 (+ E3 zero-shot evaluation on the external dataset, per fold model)
  explain    E4  Grad-CAM figure
  export     E5  ONNX FP32/INT8 size, latency and accuracy; Flask app bundle
  report         aggregate everything into results/tables, results/figures, summary.md

Usage:
  python -m ricedx.run_all --config configs/default.yaml --out results
  python -m ricedx.run_all --config configs/default.yaml --stages benchmark --models inception_v4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import data as D
from . import evaluate as E
from . import report as R
from .models import count_flops, count_params, create_model, eval_transform
from .train import fit, get_device, predict_logits

ALL_STAGES = ["leakage", "benchmark", "explain", "export", "report"]


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data(cfg, log):
    dc = cfg["data"]
    primary = D.find_dataset_root(dc["primary_root"], dc["classes"])
    if primary is None:
        raise FileNotFoundError(
            "Primary dataset not found. Attach Kaggle dataset 'bahribahri/riceleaf' "
            "or set data.primary_root in the config."
        )
    external = D.find_dataset_root(dc.get("external_root"), list(dc["external_map"]))
    log(f"primary dataset: {primary}")
    log(f"external dataset: {external if external else 'not found - E3 skipped'}")
    return primary, external


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(model, model_cfg, info, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "info": info}, path)


def load_checkpoint(path, n_classes, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = create_model(ck["model_cfg"], n_classes, False, ck["info"]["img_size"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["model_cfg"], ck["info"]


def model_cfg_by_name(cfg, name):
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise KeyError(name)


def external_metrics(logits, y):
    """Metrics on the external set: full 4-way argmax and argmax restricted to shared classes."""
    labels = sorted(set(int(v) for v in y))
    full = logits.argmax(1)
    restricted_logits = np.full_like(logits, -np.inf)
    restricted_logits[:, labels] = logits[:, labels]
    restricted = restricted_logits.argmax(1)
    out = {f"full_{k}": v for k, v in E.classification_metrics(y, full, labels).items()}
    out.update({f"restricted_{k}": v for k, v in E.classification_metrics(y, restricted, labels).items()})
    out["frac_pred_outside_shared"] = float(np.mean(~np.isin(full, labels)))
    return out


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_leakage(cfg, df, cache, out, log):
    ec = cfg["experiments"]["leakage"]
    mcfg = model_cfg_by_name(cfg, ec["model"])
    labels = list(range(len(cfg["data"]["classes"])))
    for seed in ec["seeds"]:
        for protocol in ("random", "grouped"):
            run = out / "runs" / "leakage" / f"{protocol}_seed{seed}"
            if (run.with_suffix(".json")).exists():
                log(f"skip leakage {protocol} seed {seed} (done)")
                continue
            grouped = protocol == "grouped"
            split = D.group_split if grouped else D.random_split
            tr_all, te = split(df, ec["test_fraction"], seed)
            tr, va = D.split_train_val(df, tr_all, ec["val_fraction"], seed, grouped=grouped)
            overlap = D.group_overlap_fraction(df, tr_all, te)
            log(f"leakage {protocol} seed {seed}: train {len(tr)} val {len(va)} test {len(te)} "
                f"| test images whose source is in training: {overlap:.1%}")
            model, hist, info, t_eval = fit(
                mcfg, cfg, df.iloc[tr], df.iloc[va], seed, cache, log)
            logits = predict_logits(model, df.iloc[te], t_eval, cfg, cache)
            y = df.iloc[te]["label"].to_numpy()
            m = E.classification_metrics(y, logits.argmax(1), labels, E.softmax(logits))
            m.update({"protocol": protocol, "seed": seed, "model": ec["model"],
                      "test_source_overlap": overlap, **info})
            np.savez_compressed(run.with_suffix(".npz"), test_idx=te, logits=logits)
            pd.DataFrame(hist).to_csv(out / "histories" / f"leakage_{protocol}_seed{seed}.csv", index=False)
            save_json(m, run.with_suffix(".json"))
            log(f"  -> acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f}")
            del model
            torch.cuda.empty_cache()


def stage_benchmark(cfg, df, ext_df, cache, out, log, only_models=None):
    bc = cfg["experiments"]["benchmark"]
    labels = list(range(len(cfg["data"]["classes"])))
    folds = D.group_kfold(df, bc["n_folds"], bc["seed"])
    fold_of = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_of[te] = k
    pd.DataFrame({"path": df["path"], "group": df["group"], "fold": fold_of}).to_csv(
        out / "folds.csv", index=False)
    for mcfg in cfg["models"]:
        if only_models and mcfg["name"] not in only_models:
            continue
        for k, (tr_all, te) in enumerate(folds):
            run = out / "runs" / "benchmark" / f"{mcfg['name']}_fold{k}"
            if run.with_suffix(".json").exists():
                log(f"skip {mcfg['name']} fold {k} (done)")
                continue
            D.assert_disjoint_groups(df, tr_all, te)
            tr, va = D.split_train_val(df, tr_all, bc["val_fraction"], bc["seed"] + k, grouped=True)
            D.assert_disjoint_groups(df, tr, va)
            log(f"benchmark {mcfg['name']} fold {k}: train {len(tr)} val {len(va)} test {len(te)}")
            model, hist, info, t_eval = fit(
                mcfg, cfg, df.iloc[tr], df.iloc[va], bc["seed"], cache, log)
            val_logits = predict_logits(model, df.iloc[va], t_eval, cfg, cache)
            temp = E.fit_temperature(val_logits, df.iloc[va]["label"].to_numpy())
            logits = predict_logits(model, df.iloc[te], t_eval, cfg, cache)
            y = df.iloc[te]["label"].to_numpy()
            m = E.classification_metrics(y, logits.argmax(1), labels, E.softmax(logits))
            m["ece_calibrated"] = E.expected_calibration_error(E.softmax(logits, temp), y)
            m.update({"model": mcfg["name"], "fold": k, "temperature": temp, **info,
                      "params": count_params(model),
                      "flops": count_flops(model, info["img_size"])})
            arrays = {"test_idx": te, "logits": logits}
            if ext_df is not None and cfg["experiments"].get("external", True):
                ext_logits = predict_logits(model, ext_df, t_eval, cfg, None)
                ext_y = ext_df["label"].to_numpy()
                m.update({f"ext_{a}": b for a, b in external_metrics(ext_logits, ext_y).items()})
                arrays["ext_logits"] = ext_logits
                log(f"  external: acc {m['ext_full_accuracy']:.4f} macro-F1 {m['ext_full_macro_f1']:.4f}")
            np.savez_compressed(run.with_suffix(".npz"), **arrays)
            pd.DataFrame(hist).to_csv(out / "histories" / f"{mcfg['name']}_fold{k}.csv", index=False)
            if k == 0 and mcfg["name"] in bc.get("save_checkpoints_for", []):
                save_checkpoint(model, mcfg, info, out / "checkpoints" / f"{mcfg['name']}_fold0.pt")
            save_json(m, run.with_suffix(".json"))
            log(f"  -> acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} "
                f"ECE {m['ece']:.4f}->{m['ece_calibrated']:.4f}")
            del model
            torch.cuda.empty_cache()


def _fold0_test(df, out):
    folds = pd.read_csv(out / "folds.csv")
    return df[folds["fold"].to_numpy() == 0]


def stage_explain(cfg, df, out, log):
    from .explain import gradcam_figure

    ec = cfg["experiments"]["explain"]
    ck = out / "checkpoints" / f"{ec['model']}_fold0.pt"
    if not ck.exists():
        log(f"explain: {ck} missing - run the benchmark stage first")
        return
    device = get_device()
    model, _, info = load_checkpoint(ck, len(cfg["data"]["classes"]), device)
    t_eval = eval_transform(info["img_size"], info["mean"], info["std"])
    test = _fold0_test(df, out)
    per_class = max(1, ec["n_images"] // len(cfg["data"]["classes"]))
    rows = pd.concat(
        [g.drop_duplicates("group").head(per_class) for _, g in test.groupby("label")])
    path = out / "figures" / f"gradcam_{ec['model']}.png"
    gradcam_figure(model, rows, t_eval, info["img_size"], cfg["data"]["display_names"], path, device)
    log(f"explain: wrote {path}")


def stage_export(cfg, df, out, log):
    from . import export as X

    xc = cfg["experiments"]["export"]
    labels = list(range(len(cfg["data"]["classes"])))
    test = _fold0_test(df, out)
    y = test["label"].to_numpy()
    rows = []
    for name in xc["models"]:
        ck = out / "checkpoints" / f"{name}_fold0.pt"
        if not ck.exists():
            log(f"export: {ck} missing - skipped")
            continue
        model, _, info = load_checkpoint(ck, len(labels), torch.device("cpu"))
        size = info["img_size"]
        t_eval = eval_transform(size, info["mean"], info["std"])
        xdir = out / "export"
        xdir.mkdir(parents=True, exist_ok=True)
        fp32 = X.export_onnx(model, size, xdir / f"{name}_fp32.onnx")
        calib = test.sample(min(xc["calib_images"], len(test)), random_state=0)["path"].tolist()
        int8 = xdir / f"{name}_int8.onnx"
        method = X.quantize_int8(fp32, int8, calib, t_eval)
        rec = {"model": name, "img_size": size, "params": count_params(model),
               "flops": count_flops(model, size), "quantisation": method,
               "cpu": X.cpu_name(), "threads": xc["threads"]}
        for tag, path in (("fp32", fp32), ("int8", int8)):
            med, p90 = X.latency_ms(path, size, xc["threads"], xc["latency_runs"])
            logits = X.ort_predict(path, test["path"].tolist(), t_eval, xc["threads"])
            mm = E.classification_metrics(y, logits.argmax(1), labels)
            rec.update({f"{tag}_size_mb": path.stat().st_size / 2**20,
                        f"{tag}_latency_ms_median": med, f"{tag}_latency_ms_p90": p90,
                        f"{tag}_accuracy": mm["accuracy"], f"{tag}_macro_f1": mm["macro_f1"]})
        rows.append(rec)
        log(f"export {name}: fp32 {rec['fp32_latency_ms_median']:.1f} ms, "
            f"int8 {rec['int8_latency_ms_median']:.1f} ms")
        if name == cfg["experiments"]["explain"]["model"]:
            X.write_app_bundle(fp32, out / "app_bundle", cfg["data"]["display_names"],
                               size, info["mean"], info["std"])
            log(f"export: app bundle written to {out / 'app_bundle'}")
    if rows:
        pd.DataFrame(rows).to_csv(out / "tables" / "export_metrics.csv", index=False)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--models", default="", help="comma-separated subset for the benchmark stage")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    for sub in ("runs/leakage", "runs/benchmark", "histories", "checkpoints", "figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log.txt")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")
    only_models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    log(f"config {args.config} | stages {stages} | device {get_device()} | torch {torch.__version__}")

    primary, external = resolve_data(cfg, log)
    df = D.build_index(primary, cfg["data"]["classes"])
    df.to_csv(out / "index.csv", index=False)
    log(f"indexed {len(df)} images from {df['group'].nunique()} source photographs")
    ext_df = None
    if external is not None:
        ext_df = D.build_external_index(
            external, cfg["data"]["external_map"], cfg["data"]["classes"],
            cfg["data"].get("external_max_per_class"))
        ext_df.to_csv(out / "external_index.csv", index=False)
        log(f"external set: {len(ext_df)} images")
    save_json(cfg, out / "config_used.json")

    needs_training = {"leakage", "benchmark"} & set(stages)
    cache = None
    if needs_training and cfg["data"].get("cache_primary_in_ram", False):
        log("caching primary images in RAM ...")
        cache = D.build_cache(df)

    if "leakage" in stages:
        stage_leakage(cfg, df, cache, out, log)
    if "benchmark" in stages:
        stage_benchmark(cfg, df, ext_df, cache, out, log, only_models)
    if "explain" in stages:
        stage_explain(cfg, df, out, log)
    if "export" in stages:
        stage_export(cfg, df, out, log)
    if "report" in stages:
        R.build_report(cfg, df, ext_df, out, log)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
