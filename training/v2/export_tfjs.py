"""Export the v2 Keras model to a TF.js graph model (weights in 45 MB shards).
usage: export_tfjs.py MODEL.keras OUT_DIR [converter quantization flags, default --quantize_uint8=*]
The app uses --quantize_float16=* (8-bit weights cost about 5 accuracy points)."""
import os, shutil, subprocess, sys, tempfile
import keras
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v2_layers  # noqa: F401  registers the custom layers

orig = keras.saving.load_model(sys.argv[1])
S = orig.input_shape[1]


def strip_dropout(layer):
    # Dropout layers hold a random-seed variable the TF.js converter cannot handle; they do nothing at inference
    if isinstance(layer, keras.layers.Dropout):
        return keras.layers.Identity(name=layer.name)
    if isinstance(layer, keras.Model):
        return keras.models.clone_model(layer, clone_function=strip_dropout)
    return layer.__class__.from_config(layer.get_config())


m = keras.models.clone_model(orig, clone_function=strip_dropout)
FEATURES = os.environ.get("EXPORT_FEATURES") == "1"  # export the feature extractor only (used to fit heads on quantized features)
src = {l.path: l for l in orig._flatten_layers()}
for l in m._flatten_layers():
    o = src.get(l.path)
    if o is not None and o.weights and not isinstance(o, keras.Model):
        l.set_weights(o.get_weights())
x = tf.random.uniform((2, S, S, 3), 0, 255)
diff = float(tf.reduce_max(tf.abs(orig(x, training=False) - m(x, training=False))))
assert diff < 1e-4, diff
print("inference copy matches original, max diff", diff)


@tf.function(input_signature=[tf.TensorSpec([None, S, S, 3], tf.float32, name="image")])
def serve(x):
    return {"probs": m(x, training=False)}


if FEATURES:
    feat = keras.Model(m.input, m.get_layer("features").output)

    @tf.function(input_signature=[tf.TensorSpec([None, S, S, 3], tf.float32, name="image")])
    def serve(x):
        return {"features": feat(x, training=False)}


mod = tf.Module()
mod.model = m
tmp = tempfile.mkdtemp()
tf.saved_model.save(mod, tmp, signatures={"serving_default": serve})
subprocess.run([shutil.which("tensorflowjs_converter") or "tensorflowjs_converter",
                "--input_format=tf_saved_model", "--signature_name=serving_default", *(sys.argv[3:] if len(sys.argv) > 3 else ["--quantize_uint8=*"]),
                "--weight_shard_size_bytes=45000000", tmp, sys.argv[2]], check=True)
