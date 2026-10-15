# Leakage-Aware Evaluation of InceptionV4 for Rice Leaf Disease Diagnosis and a Web-Based Decision-Support System

Reproducibility code for the study above. It uses the Kaggle dataset [`bahribahri/riceleaf`](https://www.kaggle.com/datasets/bahribahri/riceleaf): 16,000 images in four classes (bacterial leaf blight, brown spot, leaf smut, healthy). The images were produced by augmenting only **160 original photographs** (40 per class), about 100 times each. The source photograph of every image is encoded in its filename, so this repository can measure exactly how much an image-level split inflates the results.

## What the code does

| Stage | Experiment |
|---|---|
| `leakage` | **E1.** InceptionV4 trained and tested on an image-level random split (the original protocol) vs. a source-grouped split, 3 seeds. |
| `benchmark` | **E2.** Stratified *group* 5-fold cross-validation over the 160 source photographs for 7 ImageNet-pretrained models: InceptionV4, InceptionV3, ResNet-50, EfficientNet-B0, MobileNetV3-Large, ConvNeXt-Tiny, DeiT-Small. Reports accuracy, macro-F1, MCC, Cohen's κ, and ECE before and after temperature scaling. Pooled out-of-fold metrics come with 95% cluster-bootstrap CIs, exact McNemar tests (Bonferroni-adjusted) and a Friedman test. **E3.** Every fold model is also tested zero-shot on the shared classes of [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) (bacterial leaf blight, brown spot, normal). |
| `explain` | **E4.** Grad-CAM maps for InceptionV4. |
| `export` | **E5.** ONNX FP32 and static INT8 models: size, CPU latency, and accuracy after quantisation. Also writes the model bundle for the web app. |
| `report` | All tables (`results/tables`), figures (`results/figures`) and `results/summary.md`. |

## Run the full study on Kaggle (free GPU)

1. Open `notebooks/kaggle_run_all.ipynb` on Kaggle (*File → Import notebook → GitHub*), or upload it.
2. *Add Input*:
   - dataset **bahribahri/riceleaf**
   - competition **paddy-disease-classification** (join the competition and accept its rules first)
3. *Settings*: Accelerator **GPU T4 ×1** (or P100), Internet **on** (for pretrained weights and `git clone`).
4. *Save Version → Save & Run All (Commit)*. All stages are estimated to take 6–9 GPU-hours on a T4. To split the work over two sessions, see the `MODELS` / `PREVIOUS_RESULTS` options in the first code cell.
5. Download `results_for_paper.zip` (tables, figures, logs and predictions) and `app_bundle.zip` (ONNX model for the web app) from the notebook output.

## Run locally

```bash
pip install -e ".[app,dev]"
# smoke test: synthetic data, random weights, CPU, a few minutes
python scripts/make_synthetic_data.py --out smoke_data
python -m ricedx.run_all --config configs/smoke.yaml --out smoke_results
pytest -q
# full study (GPU recommended): place the datasets under ./data or set paths in configs/default.yaml
python -m ricedx.run_all --config configs/default.yaml --out results
```

Single stages or a subset of models: `--stages benchmark --models inception_v4,resnet50`. Finished runs are skipped, so an interrupted run can be resumed with the same command.

## Web application

```bash
# set RICEDX_SECRET_KEY to a long random string in production
RICEDX_MODEL_DIR=results/app_bundle python -m app     # http://127.0.0.1:8000
```

Features:

- Registration and login with hashed passwords.
- Image upload or phone-camera capture.
- Prediction with per-class probabilities.
- An "uncertain, retake the photo" answer when confidence is below `RICEDX_THRESHOLD` (default 0.70).
- Management advice for each disease, paraphrased from the IRRI Rice Knowledge Bank and the Texas Plant Disease Handbook (sources are shown in the app).
- Per-user prediction history.

## Repository layout

```
configs/      default.yaml (full study), smoke.yaml (CPU smoke test)
src/ricedx/   data.py  models.py  train.py  evaluate.py  stats.py  explain.py  export.py  report.py  run_all.py
app/          Flask app (ONNX Runtime inference), templates, knowledge.json
notebooks/    kaggle_run_all.ipynb
scripts/      make_synthetic_data.py
tests/        pytest suite (splits, metrics, models, web app)
```

## Data

The images are **not** redistributed here. See [DATA.md](DATA.md) for sources, licences and the provenance analysis.

## License

Code: MIT (see `LICENSE`). Datasets remain under their own terms.
