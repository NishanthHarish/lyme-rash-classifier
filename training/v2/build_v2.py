"""Lyme rash classifier 2.0: frozen two-backbone features + regularised linear heads.

Fine-tuning a CNN on ~970 unique rash photos overfit (about 71% test accuracy), while frozen ImageNet
features from two complementary backbones with a strongly regularised logistic-regression head
generalise better (about 79% in grouped cross-validation). So the model is:

  photo (384x384 RGB, 0-255)
    -> EfficientNetV2-S @384 global features   (L2-normalised)
    -> ResNet-50 @224 global features            (L2-normalised)
    -> scope head: is it a rash photo at all?   (logistic regression, rash vs everything else)
    -> Lyme head: erythema migrans vs other rash (logistic regression, rash photos only)
  outputs [P(other rash), P(Lyme rash), P(not a rash photo)]

Rash images are de-duplicated (exact and near-duplicate groups) and split by group, so no picture or
near-copy of it is in both train and test. Flower photos are never trained on (unseen out-of-scope probe).

usage: python build_v2.py MANIFEST SPLIT_JSON OUT_DIR
"""
import json, os, sys
from pathlib import Path
import numpy as np
import keras
import tensorflow as tf
from keras import layers, ops
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

MANIFEST, SPLIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
S = 384
C_LYME, C_GATE = 0.001, 0.01
A = keras.applications


@keras.saving.register_keras_serializable()
class L2Normalize(layers.Layer):
    def call(self, x):
        return x / ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + 1e-12)


@keras.saving.register_keras_serializable()
class CombineHeads(layers.Layer):
    """[P(not rash), P(Lyme | rash)] -> [P(other rash), P(Lyme rash), P(not a rash photo)]"""
    def call(self, x):
        gate, lyme = x
        return ops.concatenate([(1 - gate) * (1 - lyme), (1 - gate) * lyme, gate], axis=-1)


inp = layers.Input((S, S, 3), name="image")
eff = A.EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg", input_shape=(S, S, 3))
res = A.ResNet50(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
f1 = L2Normalize()(eff(inp))
f2 = L2Normalize()(res(A.resnet50.preprocess_input(layers.Resizing(224, 224)(inp))))
feats = layers.Concatenate(name="features")([f1, f2])
extractor = keras.Model(inp, feats)

items = json.load(open(MANIFEST))
split = json.load(open(SPLIT))
where = {p: k for k, ps in split.items() for p in ps}
for it in items:
    it["split"] = "unseen" if it.get("source") == "flowers" else where.get(it["path"], "unused")


def embed(paths, flip=False):
    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), (S, S))
        return tf.image.flip_left_right(img) if flip else img
    d = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=4).batch(32)
    return extractor.predict(d, verbose=0)


cache = OUT / "features.npz"
QF = os.environ.get("QUANT_FEATURES")  # dir with features from the 8-bit TF.js backbone (qfeat_all.bin, qfeat_rash_flip.bin)
if QF:
    # fit the heads on the features the deployed 8-bit browser model actually produces
    d = int(feats.shape[-1])
    X = np.fromfile(Path(QF) / "qfeat_all.bin", np.float32).reshape(-1, d)
    rash_idx = [i for i, it in enumerate(items) if it["label"] != "Other"]
    Xf = np.zeros_like(X); Xf[rash_idx] = np.fromfile(Path(QF) / "qfeat_rash_flip.bin", np.float32).reshape(-1, d)
elif cache.exists():
    z = np.load(cache); X, Xf = z["X"], z["Xf"]
else:
    X = embed([it["path"] for it in items])
    rash_idx = [i for i, it in enumerate(items) if it["label"] != "Other"]
    Xf = np.zeros_like(X); Xf[rash_idx] = embed([items[i]["path"] for i in rash_idx], flip=True)
    np.savez(cache, X=X, Xf=Xf)

lab = np.array([{"Negative": 0, "Positive": 1, "Other": 2}[it["label"]] for it in items])
sp = np.array([it["split"] for it in items]); grp = np.array([it["group"] for it in items])
src = np.array([it.get("source", "rash") for it in items])
dev = np.isin(sp, ["train", "val"]) | ((sp == "unused") & (lab == 2))  # unused = out-of-scope photos left out of v2.0 fine-tune runs
test = sp == "test"
rash = lab < 2


def fit_lyme(idx):
    Xa = np.vstack([X[idx], Xf[idx]]); ya = np.concatenate([lab[idx], lab[idx]])
    sc = StandardScaler().fit(Xa)
    return sc, LogisticRegression(C=C_LYME, max_iter=5000).fit(sc.transform(Xa), ya)


def fit_gate(idx):
    sc = StandardScaler().fit(X[idx])
    w = {0: 1.0, 1: float((lab[idx] < 2).sum() / (lab[idx] == 2).sum())}
    return sc, LogisticRegression(C=C_GATE, max_iter=5000, class_weight=w).fit(sc.transform(X[idx]), (lab[idx] == 2).astype(int))


prob = lambda m, i: m[1].predict_proba(m[0].transform(X[i]))[:, 1]

# Lyme threshold from out-of-fold predictions on the development rash photos (more stable than a small val set)
dr = np.where(dev & rash)[0]
oof = np.zeros(len(dr))
cv_scores = []
for seed in range(3):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(dr, lab[dr], grp[dr]):
        oof[te] = prob(fit_lyme(dr[tr]), dr[te])
    cv_scores.append((roc_auc_score(lab[dr], oof), ((oof >= .5) == lab[dr]).mean()))
cands = np.linspace(0.2, 0.8, 61)
J = [(oof[lab[dr] == 1] >= t).mean() + (oof[lab[dr] == 0] < t).mean() - 1 for t in cands]
THRESH = float(cands[int(np.argmax(J))])
cv_auc, cv_acc = np.mean(cv_scores, 0)
print(f"dev CV: AUC {cv_auc:.3f} acc@0.5 {cv_acc:.3f}; Lyme threshold {THRESH:.2f}")

lyme = fit_lyme(dr)
gate = fit_gate(np.where(dev)[0])

# held-out test
ti = np.where(test & rash)[0]
s = prob(lyme, ti); g = prob(gate, ti); y = lab[ti]
pred = (s >= THRESH).astype(int)
rep = classification_report(y, pred, target_names=["Negative", "Positive"], digits=3, output_dict=True)
rep_txt = classification_report(y, pred, target_names=["Negative", "Positive"], digits=3)
cm = confusion_matrix(y, pred)
ood = {}
for k in ["imagenette", "oi_scene", "oi_skin"]:
    oi = np.where(test & (src == k))[0]
    ood[k] = float((prob(gate, oi) >= .5).mean())
ui = np.where(sp == "unseen")[0]
ood["flowers_unseen"] = float((prob(gate, ui) >= .5).mean())
n_oos = int((test & (lab == 2)).sum()) + len(ui)
oos_all = float(np.mean(np.concatenate([prob(gate, np.where(test & (lab == 2))[0]), prob(gate, ui)]) >= .5))
metrics = {
    "model": "EfficientNetV2-S@384 + ResNet-50@224 frozen ImageNet features, logistic-regression heads",
    "lyme_threshold": THRESH, "dev_cv_lyme_auc": float(cv_auc), "dev_cv_accuracy_at_0.5": float(cv_acc),
    "rash_test": {"n": int(len(ti)), "accuracy": float((pred == y).mean()), "roc_auc": float(roc_auc_score(y, s)),
                  "lyme_recall": rep["Positive"]["recall"], "lyme_precision": rep["Positive"]["precision"],
                  "negative_recall": rep["Negative"]["recall"], "confusion_rows_true_Neg_Pos": cm.tolist(),
                  "wrongly_rejected_as_not_rash": float((g >= .5).mean())},
    "out_of_scope_rejection_rate": ood, "out_of_scope_overall": oos_all, "out_of_scope_n": n_oos,
    "dev_images": {"rash": int((dev & rash).sum()), "out_of_scope": int((dev & (lab == 2)).sum())},
}
print(json.dumps(metrics, indent=2)); print(rep_txt); print(cm)


def dense_from(model, name):
    """Fold StandardScaler + logistic regression into one Dense(1, sigmoid) layer."""
    sc, lr = model
    w = lr.coef_[0] / sc.scale_
    b = lr.intercept_[0] - np.dot(lr.coef_[0], sc.mean_ / sc.scale_)
    d = layers.Dense(1, activation="sigmoid", name=name)
    d.build((None, X.shape[1])); d.set_weights([w[:, None].astype("float32"), np.array([b], "float32")])
    return d


out = CombineHeads(name="probs")([dense_from(gate, "scope_head")(feats), dense_from(lyme, "lyme_head")(feats)])
model = keras.Model(inp, out, name="lyme_rash_classifier_v2")
# check the folded Keras model agrees with sklearn
chk = ti[:16]
kp = model.predict(np.stack([tf.image.resize(tf.cast(tf.io.decode_image(tf.io.read_file(items[i]["path"]), channels=3, expand_animations=False), tf.float32), (S, S)).numpy() for i in chk]), verbose=0)
if not QF:  # with 8-bit features the float Keras model differs slightly by design
    assert np.allclose(kp[:, 1] / (kp[:, 0] + kp[:, 1]), s[:16], atol=1e-3), (kp, s[:16])
model.save(OUT / "lyme_v2.keras")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "classification_report.txt").write_text(rep_txt)
np.save(OUT / "test_rash_paths.npy", np.array([items[i]["path"] for i in ti]))
