import keras
from keras import layers, ops


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


