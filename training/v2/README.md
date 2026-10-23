# Training: Lyme Rash Classifier 2.0

Version 2.0 adds an **out-of-scope check** (the model says when a photo is not a skin rash) and raises **Lyme recall from 0.67 to 0.83** on the same held-out test set.

## Results

All numbers are on held-out test photos that were never used for training or for choosing settings. Near-duplicate rash photos were grouped before splitting, so no copy of a test photo is in training.

**Rash photos (192 test images: 87 Lyme, 105 other rashes)**

| Metric | v1 recipe, same split | **v2.0** |
| --- | ---: | ---: |
| Accuracy | 75.5% | **77.6%** |
| Lyme recall (sensitivity) | 66.7% | **82.8%** |
| Lyme precision | 76.3% | 72.0% |
| Other-rash recall (specificity) | 82.9% | 73.3% |
| ROC AUC | 0.829 | **0.841** |
| Real rashes wrongly rejected as "not a rash photo" | n/a | 1.0% |

v2.0 deliberately trades some specificity for sensitivity: missing a Lyme rash is worse than a false alarm, so the decision threshold (0.44) was chosen on cross-validated development data to balance the two (Youden's J), instead of using 0.5.

**Out-of-scope photos (1,142 test images)**

| Test set | Rejected |
| --- | ---: |
| Everyday objects (Imagenette) | 100% |
| Street, home and people scenes (Open Images) | 99.4% |
| Healthy skin: hands, arms, legs, faces (Open Images crops) | 98.9% |
| Flowers, a category never seen in training (TF Flowers) | 99.8% |

The TF.js browser model (16-bit weights) gives the same predictions; this was checked by `browser_eval.py`, which runs the real web page in headless Chromium. 8-bit weights were tried and cost about 5 accuracy points (71.9%), even with the heads refitted on the 8-bit features (74.5%), so the app ships the 87 MB 16-bit model.

## Model

```
photo -> resize to 384x384 RGB (0-255)
      -> EfficientNetV2-S @384, ImageNet weights, global average pool   -> L2 normalise ┐
      -> ResNet-50 @224, ImageNet weights, global average pool          -> L2 normalise ┴ concat (3,328 features)
      -> scope head (logistic regression): P(not a rash photo)
      -> Lyme head  (logistic regression, C=0.001): P(Lyme | rash)
output [P(other rash), P(Lyme rash), P(not a rash photo)]
```

The logistic-regression heads (with their feature scaling folded in) are stored as Keras `Dense` layers, so the whole pipeline is one Keras model and one TF.js graph model (87 MB with 16-bit weights).

## What was tried

| Experiment | Rash accuracy | Rash ROC AUC | Notes |
| --- | ---: | ---: | --- |
| Frozen features + logistic regression, single backbone (grouped 5-fold CV) | 72-76% | 0.81-0.85 | EfficientNetV2-S @384 best |
| Fine-tuned EfficientNetV2-B0, 3 classes (`finetune_experiment.py`) | 71% | 0.80 | Overfits: about 775 unique rash photos is too few to fine-tune |
| **Frozen EfficientNetV2-S + ResNet-50, flip-augmented features (grouped 5-fold CV)** | **79%** | **0.87** | Chosen |
| ConvNeXt-Tiny | n/a | n/a | Dropped: too slow on CPU for this project |

## Data

- **Rash photos**: the thesis dataset (Kaggle Lyme Disease Rashes plus web images), 2,060 files. Exact duplicates removed by MD5 (987 left), then near-duplicates grouped by perceptual hash (Hamming distance 8 or less). 20 photos labelled both ways were dropped, leaving **967 unique photos** (445 Lyme, 522 other rashes) in 905 groups. Splits are 70/10/20 by group.
- **Out-of-scope photos** (4,122), built by `oi_select.py` and `build_ood.py`:
  - 1,500 Imagenette images (a 10-class ImageNet subset)
  - 900 random Open Images V6 validation scenes
  - 1,322 Open Images crops of hands, arms, legs, feet, faces and tattoos, as healthy-skin examples
  - 400 TF Flowers photos, used only for testing

The images themselves are not in this repository.

## Reproduce

Run from a working directory with the rash dataset unzipped to `data/dataset/{Positive,Negative}`.

```bash
pip install tensorflow scikit-learn imagehash pillow
# out-of-scope data
curl -O https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar xzf imagenette2-320.tgz -C ood/
curl -o ood/validation-annotations-bbox.csv https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
python oi_select.py            # choose Open Images crops and scenes, then download ood/oi_ids.txt from
                               # https://open-images-dataset.s3.amazonaws.com/validation/<id>.jpg into ood/oi_raw/
curl https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C ood/
python build_ood.py            # crops and resizes into ood/pool/
python prep.py                 # dedup + near-duplicate groups -> manifest.json
python build_v2.py manifest.json split.json runs/v2_final   # features, heads, test metrics, lyme_v2.keras
pip install tensorflowjs       # in a separate environment if it conflicts with your TensorFlow
python export_tfjs.py runs/v2_final/lyme_v2.keras ../../web-app --quantize_float16='*'
```

`split.json` is the exact train/val/test split used for the numbers above. Optional: `embed.py` and `compare_backbones.py` reproduce the backbone comparison, `baseline_v1.py split.json out.json` retrains the v1 recipe on the same split, and `browser_eval.py` scores the exported web app.

## Limitations

- Still a small dataset of web images; no skin-tone, body-site or lesion-stage labels, so performance across those groups is unknown.
- With 192 test rash photos, each accuracy figure has an uncertainty of roughly ±3 percentage points.
- Not clinically validated, and not a medical device.
