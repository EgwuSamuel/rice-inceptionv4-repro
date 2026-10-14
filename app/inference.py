"""ONNX Runtime classifier used by the web app."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


class OnnxClassifier:
    """Loads ``model.onnx`` + ``model.json`` written by ``ricedx.export``."""

    def __init__(self, bundle_dir, threads=2):
        import onnxruntime as ort

        bundle_dir = Path(bundle_dir)
        meta = json.loads((bundle_dir / "model.json").read_text())
        self.classes = meta["classes"]
        self.img_size = int(meta["img_size"])
        self.mean = np.asarray(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(meta["std"], dtype=np.float32).reshape(3, 1, 1)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(bundle_dir / "model.onnx"), so, providers=["CPUExecutionProvider"]
        )

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None].astype(np.float32)

    def predict(self, img: Image.Image) -> dict:
        logits = self.session.run(None, {"input": self.preprocess(img)})[0][0]
        z = logits - logits.max()
        probs = np.exp(z) / np.exp(z).sum()
        i = int(probs.argmax())
        return {
            "label": self.classes[i],
            "confidence": float(probs[i]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
        }
