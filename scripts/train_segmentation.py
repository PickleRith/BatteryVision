"""Train the 4-channel U-Net on the train split; select per-class thresholds on val.

    python scripts/train_segmentation.py                 # default config
    python scripts/train_segmentation.py --epochs 2      # quick smoke run

Saves models/unet_v1.pt (weights + config + thresholds) and
artifacts/metrics/unet_v1_train_log.json. The test split is not touched here.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from batteryvision import config
from batteryvision.dataset import CLASSES
from batteryvision.models import SmallUNet, soft_dice_loss
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
def predict_probs(model, images, device, bs=16):
    model.eval()
    ds = CoatingDataset(images, np.zeros(images.shape[:3] + (len(CLASSES),), bool), np.zeros((len(images), len(CLASSES))))
    out = []
    for x, _, _ in DataLoader(ds, batch_size=bs):
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy().astype(np.float16))
    return np.concatenate(out)  # (N, K, H, W)


def val_dice(probs, masks, thr):
    """Dataset-level Dice per class for given thresholds."""
    res = []
    for k in range(probs.shape[1]):
        p = probs[:, k] >= thr[k]
        g = masks[..., k]
        tp = (p & g).sum()
        d = 2 * tp + (p & ~g).sum() + (~p & g).sum()
        res.append(float(2 * tp / d) if d else float("nan"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="unet_v2")
    args = ap.parse_args()

    seed = config.load_settings()["seed"]
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = pick_device(args.device)
    size = (args.width, args.height)
    df, data = load_cached_splits(size)
    tr_img, tr_msk, tr_lab, _ = data["train"]
    va_img, va_msk, va_lab, _ = data["val"]
    print(f"device={device} train={len(tr_img)} val={len(va_img)} size={size}")

    model = SmallUNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_img) / args.batch_size)))
    loader = DataLoader(CoatingDataset(tr_img, tr_msk, tr_lab, augment=True, seed=seed),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    log, best_score, best_state = [], -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + soft_dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        probs = predict_probs(model, va_img, device)
        vd = val_dice(probs, va_msk, [0.5] * len(CLASSES))
        score = float(np.nanmean(vd))
        log.append({"epoch": ep, "train_loss": tot / len(tr_img), "val_dice@0.5": dict(zip(CLASSES, vd)),
                    "val_mean_dice@0.5": score, "seconds": time.time() - t0})
        print(f"ep {ep:2d} loss {tot / len(tr_img):.4f} val dice@0.5 "
              + " ".join(f"{c[:5]}={d:.3f}" for c, d in zip(CLASSES, vd)) + f" mean={score:.3f} ({time.time() - t0:.0f}s)")
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Threshold selection on validation only
    model.load_state_dict(best_state)
    probs = predict_probs(model, va_img, device)
    grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
    pix_thr, img_thr = [], []
    for k in range(len(CLASSES)):
        best_t, best_d = 0.5, -1.0
        for t in grid:
            d = val_dice(probs[:, k:k + 1], va_msk[..., k:k + 1], [t])[0]
            if not np.isnan(d) and d > best_d:
                best_t, best_d = float(t), d
        pix_thr.append(best_t)
        # image-level presence score = max pixel probability; threshold by val F1
        score_k = probs[:, k].reshape(len(probs), -1).max(1).astype(np.float32)
        t_img, _ = best_f1_threshold(score_k, va_lab[:, k])
        img_thr.append(t_img)
    final_vd = val_dice(probs, va_msk, pix_thr)
    print("val thresholds (pixel):", dict(zip(CLASSES, pix_thr)))
    print("val thresholds (image):", dict(zip(CLASSES, img_thr)))
    print("val dice @ chosen thr:", dict(zip(CLASSES, np.round(final_vd, 3))))

    meta = {
        "model_name": args.name,
        "architecture": f"SmallUNet(base={args.base}), {n_params:,} params, 4 sigmoid output channels",
        "input_size_wh": list(size),
        "loss": "BCEWithLogits + soft Dice (per channel, batch-level)",
        "optimizer": f"AdamW max_lr={args.lr} wd=1e-4, OneCycle, grad-norm clip={args.grad_clip}",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augmentation": "h/v flips, brightness/contrast jitter",
        "seed": seed,
        "device": str(device),
        "selected_epoch": int(np.argmax([e["val_mean_dice@0.5"] for e in log]) + 1),
        "selection_metric": "mean val Dice@0.5 across classes",
        "pixel_thresholds": dict(zip(CLASSES, pix_thr)),
        "image_thresholds": dict(zip(CLASSES, img_thr)),
        "threshold_selection": "validation split only (pixel: max dataset Dice; image: max F1 of max-pixel-prob)",
        "val_dice_at_thresholds": dict(zip(CLASSES, final_vd)),
        "manifest_sha256_16": manifest_hash(),
        "environment": environment_info(),
    }
    out = config.models_dir() / f"{args.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "meta": meta, "base": args.base}, out)
    save_json({"meta": meta, "log": log}, config.artifacts_dir() / "metrics" / f"{args.name}_train_log.json")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
