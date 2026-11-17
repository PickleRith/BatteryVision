# BatteryVision

**Electrode-coating defect inspection and analytics on public research data.**

BatteryVision analyses images of manufactured electrode coatings from the public
[CoatingVision](https://doi.org/10.1038/s41597-025-06419-1) dataset. It finds and outlines four types of visible
coating defect (surface cracks, delamination, pinholes and unclassified defects), compares a classical OpenCV
method against two learned models, and provides a Streamlit dashboard for inspecting images, exploring the dataset
and reviewing model failures.

> This is a research and portfolio prototype. It detects *visible coating defects as annotated by the dataset
> authors*. It does not predict battery safety, capacity or lifetime, and it is not based on any company's production data.

<!-- SCREENSHOT -->

## What it does

- **Inspect image:** pick a dataset image or upload one. The model's predicted defect overlay is shown next to the
  **published ground-truth annotation**, which is always labelled separately. The page also reports per-class
  probabilities, predicted pixel coverage and a demo human-review flag.
- **Dataset analytics:** label prevalence, co-occurrence, annotated mask coverage, per-run counts and split
  support. Every number is computed from the manifest.
- **Model evaluation:** held-out test metrics per class, precision-recall curves, pixel-level Dice and IoU, the
  baseline-vs-model comparison and a false-negative/false-positive gallery.
- **Methodology:** data verification, the leakage-aware split, metric definitions and limitations.

## Results (held-out test split, 352 images)

The test split holds out whole frame blocks, and all thresholds were frozen on validation. The full numbers are in
`artifacts/metrics/*_test_metrics.json`.

**Pinholes: classical baseline vs learned models**, the one class all three methods address:

| Method | Image precision | Image recall | Image F1 | Pinhole objects found | Object precision | Localizes |
|---|---|---|---|---|---|---|
| OpenCV top-hat baseline | 0.57 | 0.70 | 0.63 | 57 / 89 (0.64) | 0.32 | pinholes only |
| ResNet18 classifier | 0.77 | **0.94** | 0.85 | n/a | n/a | no |
| U-Net v2 (segmentation) | **0.91** | 0.86 | **0.88** | **73 / 89 (0.82)** | **0.86** | 4 classes |

**Per-class image-level F1** (test support in brackets):

| Defect | ResNet18 classifier | U-Net v2 |
|---|---|---|
| Surface crack (287) | **0.96** | 0.93 |
| Delamination (39) | **0.98** | 0.85 |
| Pinhole (79) | 0.85 | **0.88** |
| Unclassified (14) | **0.53** | 0.31 |

**U-Net pixel-level Dice at 640×480** (pixels summed over all test images): surface crack **0.76**,
pinhole **0.74**, delamination **0.66** (precision 0.94, recall 0.50, so regions are under-segmented),
unclassified ≈ 0.

**What this shows**
- On pinholes, the U-Net is clearly ahead of the classical baseline: it finds more pinholes (0.82 vs 0.64 object
  recall) with far fewer false candidates (0.86 vs 0.32 object precision).
- For presence alone, the classifier is better calibrated. Its review flag marks ~20% of test images, which contain
  58% of its mistakes. The U-Net's scores saturate, so its flag is uninformative.
- Unclassified, and the line between it and delamination, is the main weakness: 10 of the U-Net's 14 delamination
  false alarms are images annotated as unclassified.
- Delamination test support comes mostly from one frame block of one run, so treat those scores as optimistic.

The failure analysis and the scope of these numbers are in [artifacts/limitations.md](artifacts/limitations.md).

## Architecture

```
CoatingVision (Figshare, CC BY 4.0)
   │  scripts/inspect_dataset.py   verify pairing, mask values, channel↔label mapping (all 2,227 images)
   ▼
data/processed/manifest.csv       labels, mask coverage, run/frame groups, train/val/test split
   │
   ├── scripts/baseline_opencv.py      top-hat pinhole detector     (tuned on train+val)
   ├── scripts/train_classifier.py     ResNet18 multi-label         (thresholds from val)
   └── scripts/train_segmentation.py   4-channel U-Net              (thresholds from val)
   ▼
scripts/evaluate.py   →   artifacts/metrics/*.json|csv   (test split, scored once)
   ▼
app.py (Streamlit)    inspect · analytics · evaluation · methodology
```

## Run it

Requires Python 3.11. Commands below are for Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1. PyTorch: pick ONE
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu    # CPU
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130  # NVIDIA GPU

# 2. Everything else, plus this package in editable mode
python -m pip install -r requirements.txt
python -m pip install -e .

# 3. Data: download CoatingVision.zip (94 MB) and extract it to data/raw/CoatingVision
curl.exe -L -o data/raw/CoatingVision.zip https://ndownloader.figshare.com/files/55182476
Expand-Archive data/raw/CoatingVision.zip -DestinationPath data/raw/CoatingVision

# 4. Verify, build the manifest, run tests
python scripts/inspect_dataset.py
python scripts/prepare_data.py
python -m pytest -q

# 5. Baseline, training (GPU: ~25 min for the U-Net; CPU: several hours), evaluation
python scripts/baseline_opencv.py
python scripts/train_segmentation.py              # -> models/unet_v2.pt
python scripts/train_classifier.py                # -> models/resnet18_multilabel_v1.pt
python scripts/calibrate_thresholds.py models/unet_v2.pt   # pixel thresholds at 640x480, val only
python scripts/evaluate.py --split val
python scripts/calibrate_review.py                # review-flag band, val only
python scripts/evaluate.py                        # final test-split metrics

# 6. App
streamlit run app.py
```

Without the full dataset, the app falls back to 18 bundled test-split samples in `data/samples/`
(`python scripts/export_demo_assets.py`). Set `BATTERYVISION_DATA_ROOT` to point at the data somewhere else.

## Repository layout

```
app.py                       Streamlit app
config/settings.yaml         paths, seed, split fractions
src/batteryvision/           dataset, splits, transforms, models, inference, metrics, classical_cv, analytics, visualization
scripts/                     inspect · prepare · baseline · train · evaluate · export samples
tests/                       pairing, channel mapping, split grouping, metric edge cases, inference schema
artifacts/metrics/           saved metrics, per-image predictions, PR curves, training logs, split assignments
artifacts/methodology.md     methodology and limitations (also shown in the app)
data/processed/manifest.csv  generated manifest (versioned)
data/samples/                small CC BY 4.0 sample set for the demo
```

## Data, licence and attribution

- Data: Sampath et al., *CoatingVision: A Defect Dataset for Coating Manufacturing*, **Scientific Data** (2026),
  [doi:10.1038/s41597-025-06419-1](https://doi.org/10.1038/s41597-025-06419-1). Dataset:
  [doi:10.6084/m9.figshare.29260121.v1](https://doi.org/10.6084/m9.figshare.29260121.v1), licensed **CC BY 4.0**.
- The authors' reference implementation ([vigsam-coder/CoatingVision](https://github.com/vigsam-coder/CoatingVision))
  uses TensorFlow/Keras. This repository is an **independent PyTorch implementation** and does not reuse the authors' code.
- The raw dataset is not committed. `data/samples/` holds 18 unmodified images and masks, redistributed under CC BY 4.0 with attribution.
