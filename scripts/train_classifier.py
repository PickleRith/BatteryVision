"""Train a ResNet18 multi-label classifier (image-level presence; no localisation).

    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 1     # quick smoke run

Uses 320x240 inputs (not 224x224) so tiny pinholes are not resized away entirely.
Saves models/resnet18_multilabel_v1.pt with per-class thresholds chosen on val.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from batteryvision import config
from batteryvision.dataset import CLASSES
from batteryvision.models import build_classifier
from batteryvision.training import (
    best_f1_threshold,
    environment_info,
    load_cached_splits,
    manifest_hash,
    pick_device,
    save_json,
    set_seed,
)
from batteryvision.transforms import CoatingDataset


@torch.no_grad()
def predict(model, images, device, bs=32):
    model.eval()
    ds = CoatingDataset(images, np.zeros(images.shape[:3] + (len(CLASSES),), bool), np.zeros((len(images), len(CLASSES))))
    return np.concatenate([torch.sigmoid(model(x.to(device))).cpu().numpy() for x, _, _ in DataLoader(ds, batch_size=bs)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="resnet18_multilabel_v1")
    args = ap.parse_args()

    seed = config.load_settings()["seed"]
    set_seed(seed)
    device = pick_device(args.device)
    size = (args.width, args.height)
    _, data = load_cached_splits(size)
    tr_img, tr_msk, tr_lab, _ = data["train"]
    va_img, _, va_lab, _ = data["val"]

    # pos_weight from the TRAIN split only, capped to avoid extreme gradients
    pos = tr_lab.sum(0)
    pos_weight = np.clip((len(tr_lab) - pos) / np.maximum(pos, 1), 1.0, 10.0)
    print(f"device={device} train={len(tr_img)} val={len(va_img)} pos_weight={np.round(pos_weight, 2).tolist()}")

    model = build_classifier(pretrained=True).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(CoatingDataset(tr_img, tr_msk, tr_lab, augment=True, seed=seed),
                        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    from sklearn.metrics import average_precision_score

    log, best, best_state = [], -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, _, y in loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        pv = predict(model, va_img, device)
        aps = [float(average_precision_score(va_lab[:, k], pv[:, k])) if va_lab[:, k].any() else float("nan")
               for k in range(len(CLASSES))]
        score = float(np.nanmean(aps))
        log.append({"epoch": ep, "train_loss": tot / len(tr_img), "val_ap": dict(zip(CLASSES, aps)),
                    "val_macro_ap": score, "seconds": time.time() - t0})
        print(f"ep {ep:2d} loss {tot / len(tr_img):.4f} val AP " + " ".join(f"{c[:5]}={a:.3f}" for c, a in zip(CLASSES, aps))
              + f" macro={score:.3f} ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pv = predict(model, va_img, device)
    thr = {c: best_f1_threshold(pv[:, k], va_lab[:, k])[0] for k, c in enumerate(CLASSES)}
    print("val thresholds:", thr)
    meta = {
        "model_name": args.name,
        "architecture": "torchvision ResNet18 (ImageNet-pretrained), 4-logit head",
        "input_size_wh": list(size),
        "loss": f"BCEWithLogits, pos_weight (train split, capped at 10) = {dict(zip(CLASSES, pos_weight.round(2).tolist()))}",
        "optimizer": f"AdamW lr={args.lr} wd=1e-4, cosine",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augmentation": "h/v flips, brightness/contrast jitter",
        "seed": seed,
        "device": str(device),
        "selected_epoch": int(np.argmax([e["val_macro_ap"] for e in log]) + 1),
        "selection_metric": "macro val average precision",
        "thresholds": thr,
        "threshold_selection": "validation split only, max F1 per class",
        "manifest_sha256_16": manifest_hash(),
        "environment": environment_info(),
    }
    out = config.models_dir() / f"{args.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "meta": meta}, out)
    save_json({"meta": meta, "log": log}, config.artifacts_dir() / "metrics" / f"{args.name}_train_log.json")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
