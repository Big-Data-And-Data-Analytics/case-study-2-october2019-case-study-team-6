import tensorflow as tf
from tensorflow import keras

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print(tf.config.list_physical_devices("GPU"))

print("Tensorflow built with cuda")
print(tf.test.is_built_with_cuda())

print("Tensorflow version")
print(tf.version.VERSION)

import sys
print(sys.version)

print(tf.reduce_sum(tf.random.normal([1000,1000])))
