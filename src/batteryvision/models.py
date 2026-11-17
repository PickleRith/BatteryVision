"""Model definitions: a small 4-channel U-Net and a ResNet18 multi-label classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Classic 4-level U-Net with a configurable width. Outputs one logit map per class
    (independent sigmoid channels, because defects can overlap)."""

    def __init__(self, in_ch: int = 3, n_classes: int = len(CLASSES), base: int = 16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([_block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])])
        self.bottleneck = _block(c[3], c[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, s], 1))
        return self.head(x)


def build_classifier(pretrained: bool = True, n_classes: int = len(CLASSES)) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-channel soft Dice computed over the whole batch, averaged over channels."""
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()
