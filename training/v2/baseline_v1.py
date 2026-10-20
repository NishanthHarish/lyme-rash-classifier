"""v1 recipe (ResNet-50, 2 classes, conv5 fine-tune) retrained on the v2 rash split, for a fair comparison.
usage: baseline_v1.py SPLIT_JSON OUT_JSON"""
import json, sys, numpy as np, tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, roc_auc_score
split = json.load(open(sys.argv[1]))
lab = lambda p: 1 if "/Positive/" in p else (0 if "/Negative/" in p else None)
rows = {k: [(p, lab(p)) for p in v if lab(p) is not None] for k, v in split.items()}
IMG, BS = 224, 32
def load(path, label):
    img = tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), (IMG, IMG), method="nearest")
    return tf.cast(img, tf.float32), tf.one_hot(label, 2)
augment = tf.keras.Sequential([layers.RandomRotation(25 / 360, fill_mode="nearest"), layers.RandomZoom(0.1, fill_mode="nearest"),
                               layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"), layers.RandomFlip("horizontal")])
def ds(r, training):
    d = tf.data.Dataset.from_tensor_slices(([a for a, _ in r], [b for _, b in r]))
    if training: d = d.shuffle(len(r), seed=42)
    d = d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BS)
    if training: d = d.map(lambda x, y: (augment(x, training=True), y))
    return d.prefetch(tf.data.AUTOTUNE)
inputs = layers.Input(shape=(IMG, IMG, 3))
base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
h = layers.AveragePooling2D(pool_size=(7, 7))(base(tf.keras.applications.resnet50.preprocess_input(inputs), training=False))
h = layers.Dense(64, activation="relu")(layers.Flatten()(h)); h = layers.Dropout(0.5)(h)
model = Model(inputs, layers.Dense(2, activation="softmax")(h)); base.trainable = False
stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["accuracy"])
model.fit(ds(rows["train"], True), validation_data=ds(rows["val"], False), epochs=30, callbacks=[stop], verbose=2)
base.trainable = True
for l in base.layers: l.trainable = l.name.startswith("conv5_")
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-5), metrics=["accuracy"])
model.fit(ds(rows["train"], True), validation_data=ds(rows["val"], False), epochs=15, callbacks=[stop], verbose=2)
p = model.predict(ds(rows["test"], False), verbose=0)[:, 1]; y = np.array([b for _, b in rows["test"]])
r = classification_report(y, (p >= .5).astype(int), output_dict=True, digits=3)
out = {"accuracy": float(((p >= .5) == y).mean()), "lyme_recall": r["1"]["recall"], "lyme_precision": r["1"]["precision"],
       "negative_recall": r["0"]["recall"], "roc_auc": float(roc_auc_score(y, p)), "n": int(len(y))}
print(out); json.dump(out, open(sys.argv[2], "w"), indent=2)
