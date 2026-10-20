import json, sys, numpy as np, tensorflow as tf
from tensorflow import keras
A=keras.applications
name=sys.argv[1]; FLIP=len(sys.argv)>2
M={'resnet50':(A.ResNet50,A.resnet50.preprocess_input,224),
   'effv2b0':(A.EfficientNetV2B0,lambda x:x,224),
   'effv2b2':(A.EfficientNetV2B2,lambda x:x,260),
   'effv2s':(A.EfficientNetV2S,lambda x:x,384),
   }
fn,pre,S=M[name]
base=fn(include_top=False,weights='imagenet',pooling='avg',input_shape=(S,S,3))
items=json.load(open('manifest.json'))
if FLIP: items=[i for i in items if i['label']!='Other']
def load(p):
    img=tf.io.decode_image(tf.io.read_file(p),channels=3,expand_animations=False)
    x=tf.image.resize(tf.cast(img,tf.float32),(S,S))
    return pre(tf.image.flip_left_right(x) if FLIP else x)
ds=tf.data.Dataset.from_tensor_slices([i['path'] for i in items]).map(load,num_parallel_calls=4).batch(32)
E=base.predict(ds,verbose=0)
np.save(f'emb_{name}{"_flip" if FLIP else ""}.npy',E); print(name,E.shape)
