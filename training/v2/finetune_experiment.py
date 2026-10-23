"""Lyme rash classifier 2.0 training.

Three classes: Negative (other rash), Positive (erythema migrans / Lyme), Other (not a rash photo).
Rash images are de-duplicated (exact + near-duplicate groups) and split by group, so no picture
or near-copy of it is in both train and test. Flower photos are never trained on and serve as an
unseen out-of-scope probe.

usage: python train_v2.py MANIFEST OUT_DIR BACKBONE
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

MANIFEST, OUT, BACKBONE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["Negative", "Positive", "Other"]
IMG = int(sys.argv[5]) if len(sys.argv) > 5 else 224
BS = 32
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
A = keras.applications
BACKBONES = {"convnext_tiny": A.ConvNeXtTiny, "convnext_small": A.ConvNeXtSmall,
             "effv2b0": A.EfficientNetV2B0, "effv2b2": A.EfficientNetV2B2, "effv2s": A.EfficientNetV2S,
             "resnet50": A.ResNet50}

items = json.load(open(MANIFEST))
for it in items:
    it["y"] = CLASSES.index(it["label"])

# group-level split: 70% train / 10% val / 20% test, stratified by class
groups = defaultdict(list)
for it in items:
    if it.get("source") != "flowers":
        groups[(it["label"], it["group"])].append(it)
split = {"train": [], "val": [], "test": []}
for label in CLASSES:
    keys = sorted(k for k in groups if k[0] == label)
    random.shuffle(keys)
    n = len(keys); a, b = int(n * .7), int(n * .8)
    for name, ks in (("train", keys[:a]), ("val", keys[a:b]), ("test", keys[b:])):
        for k in ks:
            split[name] += groups[k]
unseen = [it for it in items if it.get("source") == "flowers"]
# out-of-scope photos are easy to separate; keep training balanced by using at most 1,200 of them
oos_train = [it for it in split["train"] if it["label"] == "Other"]
random.shuffle(oos_train)
split["train"] = [it for it in split["train"] if it["label"] != "Other"] + oos_train[:1200]
for k, v in split.items():
    print(k, {c: sum(it["label"] == c for it in v) for c in CLASSES})


def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(tf.cast(img, tf.float32), (IMG, IMG))
    img.set_shape((IMG, IMG, 3))
    return img, tf.one_hot(label, len(CLASSES))


augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.08, fill_mode="reflect"),
    layers.RandomZoom((-0.2, 0.1), fill_mode="reflect"),
    layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
    layers.RandomBrightness(0.15, value_range=(0, 255)),
    layers.RandomContrast(0.2),
])


def ds(rows, training):
    d = tf.data.Dataset.from_tensor_slices(([r["path"] for r in rows], [r["y"] for r in rows]))
    if training:
        d = d.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)
    d = d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BS)
    if training:
        d = d.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0, 255), y),
                  num_parallel_calls=tf.data.AUTOTUNE)
    return d.prefetch(tf.data.AUTOTUNE)


# model takes raw 0-255 RGB; ConvNeXt / EfficientNetV2 include their own normalisation
inputs = layers.Input((IMG, IMG, 3))
base = BACKBONES[BACKBONE](include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3), pooling="avg")
base.trainable = False
x = A.resnet50.preprocess_input(inputs) if BACKBONE == "resnet50" else inputs
x = base(x, training=False)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax", name="probs")(x)
model = keras.Model(inputs, outputs)

counts = np.bincount([r["y"] for r in split["train"]], minlength=3)
class_weight = {i: float(counts.sum() / (3 * c)) for i, c in enumerate(counts)}
print("class weights", class_weight)
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
train_ds, val_ds = ds(split["train"], True), ds(split["val"], False)
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
H1 = model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weight, callbacks=[stop], verbose=2)

# stage 2: fine-tune the last part of the backbone at a low learning rate
base.trainable = True
n_layers = len(base.layers)
for layer in base.layers[: int(n_layers * 0.6)]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
EPOCHS2 = int(sys.argv[4]) if len(sys.argv) > 4 else 25
sched = keras.optimizers.schedules.CosineDecay(1e-4, EPOCHS2 * len(train_ds), warmup_target=None)
model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=0.05), loss=loss, metrics=["accuracy"])
stop2 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
H2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2, class_weight=class_weight, callbacks=[stop2], verbose=2)


def predict(rows):
    return model.predict(ds(rows, False), verbose=0)


def lyme_score(p):  # P(Lyme | it is a rash photo)
    return p[:, 1] / np.clip(p[:, 0] + p[:, 1], 1e-7, None)


# pick the Lyme threshold on validation: highest Youden J, i.e. best balance of recall and specificity
pv = predict(split["val"]); yv = np.array([r["y"] for r in split["val"]])
rv = yv < 2; sv = lyme_score(pv[rv]); tv = yv[rv]
cands = np.linspace(0.05, 0.95, 91)
J = [((sv[tv == 1] >= t).mean() + (sv[tv == 0] < t).mean() - 1) for t in cands]
THRESH = float(cands[int(np.argmax(J))])
print("Lyme threshold (val Youden):", THRESH)

pt = predict(split["test"]); yt = np.array([r["y"] for r in split["test"]])
src = np.array([r.get("source", "rash") for r in split["test"]])
rash = yt < 2
gate_rash = pt[rash].argmax(1) == 2                 # real rash photos wrongly turned away
s = lyme_score(pt[rash]); yr = yt[rash]
pred = (s >= THRESH).astype(int)
rep = classification_report(yr, pred, target_names=CLASSES[:2], digits=3, output_dict=True)
rep_txt = classification_report(yr, pred, target_names=CLASSES[:2], digits=3)
cm = confusion_matrix(yr, pred)
auc = roc_auc_score(yr, s)
# end-to-end: what the app shows for each rash photo (rejected counts as an error)
e2e = np.where(gate_rash, -1, pred)
pf = predict(unseen)
ood = {k: float((pt[src == k].argmax(1) == 2).mean()) for k in sorted(set(src) - {"rash"})}
ood["flowers_unseen"] = float((pf.argmax(1) == 2).mean())
summary = {
    "backbone": BACKBONE, "image_size": IMG, "classes": CLASSES, "lyme_threshold": THRESH,
    "split_sizes": {k: len(v) for k, v in split.items()}, "unseen_ood": len(unseen),
    "rash_test": {"n": int(rash.sum()), "accuracy": float((pred == yr).mean()), "roc_auc": float(auc),
                  "lyme_recall": rep["Positive"]["recall"], "lyme_precision": rep["Positive"]["precision"],
                  "negative_recall": rep["Negative"]["recall"], "confusion_rows_true_Neg_Pos": cm.tolist(),
                  "wrongly_rejected_as_not_rash": float(gate_rash.mean()),
                  "end_to_end_accuracy": float((e2e == yr).mean())},
    "out_of_scope_rejection_rate": ood,
    "history": {k: [float(v) for v in H1.history[k] + H2.history[k]] for k in H1.history},
}
print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
print(rep_txt); print(cm)
model.save(OUT / "lyme_v2.keras")
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
(OUT / "classification_report.txt").write_text(rep_txt)
json.dump({k: [r["path"] for r in v] for k, v in split.items()}, open(OUT / "split.json", "w"))
