"""Gradio app for the Lyme disease (erythema migrans) rash classifier."""
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).parent / "lyme_model.keras"
CLASSES = ["Negative", "Positive"]
IMG = 224

model = tf.keras.models.load_model(MODEL_PATH)


def classify(image):
    if image is None:
        return {}
    # same input the model was trained on: RGB, 224x224 nearest resize, 0-255 floats
    x = tf.image.resize(np.asarray(image.convert("RGB")), (IMG, IMG), method="nearest")
    probs = model.predict(tf.cast(x, tf.float32)[None], verbose=0)[0]
    return {f"Lyme {c.lower()}": float(p) for c, p in zip(CLASSES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Photo of the skin rash"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Lyme Disease Rash Classifier",
    description=(
        "Upload a photo of a skin rash to check whether it looks like the erythema migrans "
        "(bullseye) rash of Lyme disease. ResNet50 model trained on about 1,000 rash images.\n\n"
        "**This is a research demo, not a medical diagnosis.** See a doctor about any suspected tick bite or rash."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
