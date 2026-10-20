# Lyme Disease Rash Classifier

A ResNet-50 deep learning model that estimates whether a skin rash photo looks like erythema migrans, the "bullseye" rash of early Lyme disease.

**[Try the live demo](https://huggingface.co/spaces/nishanthharish/lyme-rash-classifier)** — the model runs entirely in your browser with TensorFlow.js, so photos are never uploaded.

> **Not a medical device.** This is a research demo. It cannot diagnose Lyme disease, and a negative result does not rule it out. If you have a suspected tick bite or an expanding rash, see a doctor.

## About

Built from my MSc Data Analytics thesis at Dublin Business School, *Predictive Modeling of Lyme Disease Detection Using Deep Learning* (supervisor: Meghana Kshirsagar, January 2024). Research question: how effective is fine-tuning a ResNet-50 model at classifying Lyme disease from dermatological images?

## Repository layout

| Folder | Contents |
| --- | --- |
| `notebook/` | Original thesis notebook (Google Colab) |
| `training/` | `train.py` retraining script, test metrics and classification report |
| `web-app/` | The deployed in-browser app: `index.html`, bundled `tf.min.js`, TensorFlow.js model (`model.json` + 23 MB quantized weights) |
| `gradio-app/` | Alternative Python/Gradio app (needs the full Keras model, not included) |

## Data

- Kaggle [Lyme Disease Rashes](https://www.kaggle.com/datasets/sshikamaru/lyme-disease-rashes) dataset plus web-sourced images: 2,060 files (978 Positive, 1,082 Negative). The images are not included in this repo.
- Most images appeared twice (`x.jpg` and `x copy.jpg`). Duplicates were removed by content hash before splitting, leaving **987 unique images**: 711 train, 78 validation, 198 test, so no test image leaks into training.
- Images resized to 224 x 224 RGB; training augmentation with rotation (±25°), zoom and shift (±10%) and horizontal flips.

## Model

- ResNet-50 pretrained on ImageNet, with a head of AveragePooling2D (7x7), Flatten, Dense 64 ReLU, Dropout 0.5 and Dense 2 softmax.
- Stage 1: frozen base, head trained with Adam (lr 1e-4), batch 32, early stopping.
- Stage 2: the last residual block (`conv5`) fine-tuned at lr 1e-5.
- Converted to a TensorFlow.js graph model with 8-bit quantized weights (23 MB).

## Results (198 held-out test images)

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Positive (Lyme) | 0.87 | 0.60 | 0.71 |
| Negative | 0.69 | 0.91 | 0.78 |

**Accuracy: 75.3%.** When the model says "Lyme" it is usually right, but it misses about 4 in 10 Lyme rashes. The thesis reported 80.8%, but that figure was inflated by duplicate images appearing in both training and test; 75.3% is the honest number on a deduplicated split. The quantized browser model scores the same as the original Keras model.

## Run it yourself

**Web app:** serve the `web-app/` folder with any static server and open it in a browser:

    cd web-app
    python -m http.server 8000

Then open http://localhost:8000.

**Retrain:** download the dataset into a folder with `Negative/` and `Positive/` subfolders, then:

    pip install tensorflow scikit-learn
    python training/train.py path/to/dataset output_dir improved

## Limitations

- Low recall on Lyme rashes is the main weakness.
- Small dataset (987 images) of web photos with varied lighting and quality.
- No labels for skin tone, lesion stage or body site, so performance across those groups is unknown.
- Not clinically validated.

## Future work

Raise Positive recall (threshold tuning, class weighting, stronger backbones), collect more diverse clinically labelled images, add Grad-CAM explanations, and validate with clinicians.

## Tech stack

Python, TensorFlow / Keras, scikit-learn, Google Colab, TensorFlow.js, Hugging Face Spaces.

## References

- Hossain, S.I. et al. (2021). Benchmarking convolutional neural networks for diagnosing Lyme disease from images. arXiv:2106.14465.
- Hossain, S.I. et al. (2022). Exploring convolutional neural networks with transfer learning for diagnosing Lyme disease from skin lesion images. *Computer Methods and Programs in Biomedicine*, 215, 106624.
- He, K. et al. (2016). Deep residual learning for image recognition. CVPR.

Author: Nishanth Harish
