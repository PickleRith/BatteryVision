# Model weights

Checkpoints are not committed by default (`*.pt` is gitignored). Recreate them with:

```powershell
python scripts/train_segmentation.py                       # models/unet_v2.pt  (~8 MB, ~30 min on an RTX 3050 Ti)
python scripts/calibrate_thresholds.py models/unet_v2.pt   # val-only pixel thresholds at 640x480
python scripts/train_classifier.py                         # models/resnet18_multilabel_v1.pt (~45 MB, ~10 min)
python scripts/evaluate.py --split val; python scripts/calibrate_review.py
```

Each checkpoint stores its weights, its training configuration, its validation-chosen thresholds, its review band
and the hash of the manifest it was trained on (`meta`). `archive/unet_v1.pt` is the earlier U-Net whose validation
scores collapsed during training; it is kept for reference only.
