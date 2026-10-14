"""ONNX export, INT8 static quantisation and CPU latency measurement."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_rgb


def export_onnx(model, img_size, path):
    model = model.cpu().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, x, str(path), dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5 has no ``dynamo`` argument
        torch.onnx.export(model, x, str(path), **kwargs)
    return Path(path)


def _preprocess(paths, transform):
    return np.stack([transform(load_rgb(p)).numpy() for p in paths]).astype(np.float32)


def quantize_int8(fp32_path, int8_path, calib_paths, transform):
    """Static QDQ INT8 quantisation calibrated on real images; dynamic as fallback."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    batch = _preprocess(calib_paths, transform)

    # shape inference + graph optimisation recommended by ONNX Runtime before quantising
    source = Path(fp32_path)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        prepped = source.with_name(source.stem + "_prep.onnx")
        quant_pre_process(str(source), str(prepped))
        source = prepped
    except Exception as exc:  # pragma: no cover - optional step
        print(f"  quantisation pre-processing skipped ({exc})")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": batch[i : i + 1]} for i in range(len(batch))])

        def get_next(self):
            return next(self.it, None)

    try:
        quantize_static(
            str(source), str(int8_path), Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        return "static_qdq"
    except Exception as exc:  # pragma: no cover - depends on ORT build
        print(f"  static quantisation failed ({exc}); using dynamic")
        quantize_dynamic(str(source), str(int8_path), weight_type=QuantType.QInt8)
        return "dynamic"


def ort_session(path, threads):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def latency_ms(path, img_size, threads, runs, warmup=5):
    sess = ort_session(path, threads)
    x = np.random.default_rng(0).standard_normal((1, 3, img_size, img_size)).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def ort_predict(path, paths, transform, threads, batch=32):
    sess = ort_session(path, threads)
    out = []
    for i in range(0, len(paths), batch):
        out.append(sess.run(None, {"input": _preprocess(paths[i : i + batch], transform)})[0])
    return np.concatenate(out)


def write_app_bundle(onnx_path, out_dir, class_names, img_size, mean, std):
    """Model + metadata consumed by the Flask app."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model.onnx"
    target.write_bytes(Path(onnx_path).read_bytes())
    meta = {"classes": class_names, "img_size": img_size, "mean": mean, "std": std}
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2))
    return target


def cpu_name():
    name = platform.processor() or platform.machine()
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return name
