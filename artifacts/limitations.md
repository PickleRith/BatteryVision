**Observed failure modes (U-Net v2, test split):**

1. **Pinholes cut off at the image border are missed.** The three most confident pinhole misses (image_251, image_1850,
   image_2208) are the same half-visible bright spot on the left edge of neighbouring frames from run R1-600um.
   The patch boundary removes most of the round shape the model relies on. Overall the U-Net finds 73/89 annotated
   pinholes (object recall 0.82). Tiling with overlap, or giving the model context beyond the patch, would be the next thing to test.
2. **Small bright raised features: delamination vs unclassified.** 13 of the 14 delamination false alarms come from
   run R1-600um, which has no delamination labels at all, and **10 of the 14 are images the authors annotated as
   *unclassified***. The model sees a small bright raised mark (e.g. image_783, image_1562, image_102) and calls it
   delamination. With 52 unclassified training images, the boundary between these two classes is not learnt.
   Unclassified is effectively unsupported: pixel Dice ≈ 0, image-level F1 0.31.
3. **Delamination is under-segmented.** When the model finds delamination, it outlines only about half the annotated
   region (pixel precision 0.94, recall 0.50).
4. **Crack predictions on crack-free images.** 68% of test images without annotated cracks still receive some
   predicted crack pixels, mostly short, faint, line-like texture. At image level this gives 41 crack false alarms among the 65
   crack-free test images.
5. **U-Net scores are saturated, not calibrated.** The image score (highest pixel probability per class) piles up at
   a few values. For example, 80% of positive pinhole test images score exactly 0.618. This lowers threshold-free metrics
   (pinhole AP 0.89 vs the classifier's 0.97), and it makes the review flag uninformative for the U-Net: it caught 13%
   of test images with errors, against 58% for the classifier's flag at a ~20% flag rate.

**Why this is not production-certified:**

- **Evaluation scope.** The test split holds out frame blocks, not whole runs or new production conditions, so
  these numbers do not show the model generalises to a new coating line, camera or lighting set-up.
- **Concentrated test labels.** Delamination test support (39 images) comes mostly from one frame block of one run
  (28 images), and unclassified has only 14 test images, so those scores have wide uncertainty.
- **How the test split was used.** The U-Net's test split was scored more than once. The first scoring revealed
  that pixel thresholds had been chosen at training resolution, which over-segmented at evaluation resolution. Thresholds were then
  re-chosen **on validation only**, at full resolution (`scripts/calibrate_thresholds.py`), and test was re-scored. No
  test result was used to choose a threshold, model or epoch. U-Net v1 was replaced by v2 because v1's
  validation scores collapsed during training (epoch 21); v2 adds gradient clipping and a lower learning rate.
- **Labels are the reference.** Annotations come from the dataset authors and have not been independently reviewed.
  Coverage is a share of pixels, not a calibrated physical area.
- **No acceptance criteria.** The review flag is a demo heuristic. Nothing here is a pass/fail rule, and nothing
  says anything about battery or cell performance.
