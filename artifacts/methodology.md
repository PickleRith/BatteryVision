### What the data is

[CoatingVision](https://doi.org/10.1038/s41597-025-06419-1) (Sampath et al., *Scientific Data*, 2026) is a
public dataset of **2,227 RGB patches (640×480)** of manufactured electrode coatings, relevant to batteries
and fuel cells. Each patch carries image-level labels for four defect types (**surface crack, delamination,
pinhole, unclassified**) and a 4-channel segmentation mask. A patch can carry several labels.

It is **not** data from any company's production line, and this app does **not** predict battery health,
capacity, safety or lifetime. It detects *visible coating defects as annotated by the dataset authors*.

### What was verified before modelling

`scripts/inspect_dataset.py` checks every file. Results on the downloaded archive (Figshare v1):

- every one of the 2,227 labelled images has an image and a mask of identical size; masks are RGBA with values {0, 255};
- **mask channel mapping:** read with Pillow in RGBA order, channel *k* is non-empty exactly when the *k*-th
  label column is 1, for all four classes and all 2,227 images (zero disagreements). Order: R = surface crack,
  G = delamination, B = pinhole, A = unclassified. (OpenCV loads BGRA and would silently swap crack and pinhole,
  so masks are always read through `load_mask`.)
- 44 images have no defect label and empty masks. They are reported as **"no annotated defect"**, never as "clean" or "pass".

### Annotation vs prediction

- **Published annotation / ground truth:** masks and labels from the dataset authors. Shown only for dataset images, always titled as such.
- **Model prediction:** output of a model trained in this repository. Always shown with the model name.
- The image classifier produces **presence probabilities only**. It never draws masks or reports coverage.

### Splitting without leakage

Patches are cut from video frames of **8 acquisition runs** (run, frame and patch indices are encoded in
`original_file_name`). Up to 9 patches come from one frame, and neighbouring frames look nearly identical, so a
random image split would leak. Images are grouped by **(run, frame // 50)** and whole groups are assigned to
train/val/test (target 70/15/15) by a greedy multi-label stratifier that places groups with rare classes first
(`seed = 42`). Split assignments are saved in `artifacts/metrics/split_assignments.csv`.

Limitations of this split:
- frames just either side of a block boundary can still be similar;
- delamination is concentrated in a few runs, so validation and test delamination each come largely from one
  frame block. Delamination scores therefore have high variance and limited generality;
- no run is held out entirely, so these numbers do **not** measure generalisation to a new coating run.

### Models

| | Method | Output |
|---|---|---|
| Baseline | White top-hat → robust threshold → area/circularity filter (OpenCV) | pinhole candidates only |
| Classifier | ResNet18 (ImageNet-pretrained), 4 sigmoid outputs, BCE with train-split `pos_weight` | image-level presence |
| Segmentation | Small 4-level U-Net, 4 independent sigmoid channels, BCE + soft Dice | per-pixel masks for 4 classes |

Both learned models train at 320×240 (half resolution). Masks are downsampled with an *any-pixel* rule so 1–2 px
cracks and pinholes survive, and evaluation is done at the original 640×480 resolution. Per-class thresholds are
chosen on the **validation split only**. U-Net pixel thresholds are re-selected on validation at 640×480
(`scripts/calibrate_thresholds.py`), because thresholds chosen at training resolution over-segmented at evaluation
resolution. The model checkpoint was selected by validation Dice.

U-Net v1 (max learning rate 1e-3, no clipping) collapsed on validation at epoch 21: three classes dropped to zero
Dice while training loss kept falling. v2 (max learning rate 5e-4, gradient-norm clipping 1.0) trained stably and was
selected on validation. Both training logs are in `artifacts/metrics/`.

### Metric definitions

- *Image level:* precision / recall / F1 per class at the frozen threshold, plus average precision (threshold-free).
  For the U-Net, the image score is the class channel's highest pixel probability.
- *Pixel level:* Dice, IoU, pixel precision and recall from TP/FP/FN pixels summed over **all** test images,
  including images where the class is absent. "Mean per-image Dice" averages only images where the class is
  annotated. "Predicted on images without class" is the share of such images with any predicted pixel.
- *Pinhole objects:* an annotated pinhole counts as found if a predicted pixel lies within 3 px of it. The same rule is used for the baseline.
- *Coverage* is the percentage of image pixels. It is not a physical area, because no calibrated scale is published.

### Review flag

`review_required` is a **demo heuristic**: it flags an image when any class score lies within a band of that
class's threshold. Each model's band is set on validation so that about 15% of validation images are flagged
(`scripts/calibrate_review.py`). On test, the classifier's flag covers 20% of images and 58% of the images it gets
wrong. The U-Net's flag is uninformative because its scores saturate. The flag has not been validated as an
industrial acceptance rule.

### Attribution

Data: Sampath et al., CoatingVision, *Scientific Data* (2026), [doi:10.1038/s41597-025-06419-1](https://doi.org/10.1038/s41597-025-06419-1);
dataset [doi:10.6084/m9.figshare.29260121.v1](https://doi.org/10.6084/m9.figshare.29260121.v1), CC BY 4.0.
The authors' reference code ([github.com/vigsam-coder/CoatingVision](https://github.com/vigsam-coder/CoatingVision))
uses TensorFlow/Keras. This project is an **independent PyTorch implementation** and is not the authors' code.
