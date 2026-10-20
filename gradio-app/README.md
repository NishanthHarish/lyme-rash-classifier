# Gradio version

A Python/Gradio app for the same classifier. It needs the full Keras model `lyme_model.keras` (207 MB), which is too big for GitHub and is not in this repo. Produce it with `training/train.py` and place it next to `app.py`.

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:7860.
