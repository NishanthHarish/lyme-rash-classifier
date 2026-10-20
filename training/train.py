"""Retrain the Lyme classifier from nishanth_lyme.ipynb.

Same model, input size, preprocessing (raw 0-255 RGB, no preprocess_input),
optimizer, LR, batch size and epochs as the notebook. One change: the dataset
holds most images twice ("x.jpg" and "x copy.jpg"), so images are de-duplicated
by content before splitting, keeping test images out of training.
"""
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Negative", "Positive"]
IMG = 224
BS = 32
EPOCHS = 10
INIT_LR = 1e-4
# "improved": ImageNet preprocessing inside the model, longer training, then fine-tune the last ResNet block
IMPROVED = len(sys.argv) > 3 and sys.argv[3] == "improved"

# de-duplicate by content; drop images labelled with both classes
by_hash = defaultdict(set)
first_path = {}
for label in CLASSES:
    for p in sorted((DATA / label).glob("*.jpg")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        by_hash[h].add(label)
        first_path.setdefault(h, p)
items = [(str(first_path[h]), CLASSES.index(next(iter(ls))))
         for h, ls in by_hash.items() if len(ls) == 1]
items.sort()
print(f"unique images: {len(items)} (dropped {sum(len(l) > 1 for l in by_hash.values())} with conflicting labels)")

# notebook split: 80% train+val (10% of that is val), 20% test
random.seed(42)
random.shuffle(items)
i = int(len(items) * 0.8)
trainval, test = items[:i], items[i:]
j = int(len(trainval) * 0.1)
val, train = trainval[:j], trainval[j:]
print(f"train {len(train)}  val {len(val)}  test {len(test)}")


def load(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG, IMG), method="nearest")  # flow_from_directory default
    return tf.cast(img, tf.float32), tf.one_hot(label, len(CLASSES))


augment = tf.keras.Sequential([
    layers.RandomRotation(25 / 360, fill_mode="nearest"),
    layers.RandomZoom(0.1, fill_mode="nearest"),
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    layers.RandomFlip("horizontal"),
])


def ds(rows, training):
    d = tf.data.Dataset.from_tensor_slices(([r[0] for r in rows], [r[1] for r in rows]))
    if training:
        d = d.shuffle(len(rows), seed=42)
    d = d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BS)
    if training:
        d = d.map(lambda x, y: (augment(x, training=True), y))
    return d.prefetch(tf.data.AUTOTUNE)


inputs = layers.Input(shape=(IMG, IMG, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs) if IMPROVED else inputs
base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
head = layers.AveragePooling2D(pool_size=(7, 7))(base(x, training=False))
head = layers.Flatten(name="flatten")(head)
head = layers.Dense(64, activation="relu")(head)
head = layers.Dropout(0.5)(head)
head = layers.Dense(len(CLASSES), activation="softmax")(head)
model = Model(inputs=inputs, outputs=head)
base.trainable = False

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
              metrics=["accuracy"])
if not IMPROVED:
    H = model.fit(ds(train, True), validation_data=ds(val, False), epochs=EPOCHS, verbose=2)
else:
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    H = model.fit(ds(train, True), validation_data=ds(val, False), epochs=30, callbacks=[stop], verbose=2)
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith("conv5_")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  metrics=["accuracy"])
    H2 = model.fit(ds(train, True), validation_data=ds(val, False), epochs=15, callbacks=[stop], verbose=2)
    for k, v in H2.history.items():
        H.history[k] += v

probs = model.predict(ds(test, False), verbose=0)
pred = probs.argmax(1)
y = np.array([r[1] for r in test])
report = classification_report(y, pred, target_names=CLASSES, digits=3)
cm = confusion_matrix(y, pred)
print(report)
print(cm)

model.save(OUT / "lyme_model.keras")
(OUT / "metrics.json").write_text(json.dumps({
    "preprocessing": "resnet50.preprocess_input inside model" if IMPROVED else "raw 0-255",
    "unique_images": len(items), "train": len(train), "val": len(val), "test": len(test),
    "test_accuracy": float((pred == y).mean()),
    "confusion_matrix_rows_true_Negative_Positive": cm.tolist(),
    "history": {k: [float(v) for v in vs] for k, vs in H.history.items()},
}, indent=2))
(OUT / "classification_report.txt").write_text(report)
