import logging
import numpy as np
import matplotlib.pyplot as plt
import time

from plugins.plugin_loader import PluginLoader
from lib.utils import get_folder, get_image_paths

import tensorflow as tf


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# global variables
num_gpu = 1
batch_size = 4
input_shape = (128, 128, 3)
output_shape = (128, 128, 3)
trainer_name = 'dfl-h128'
dir_a = '/home/ubuntu/faceswap/data/faces/trump/*'
dir_b = '/home/ubuntu/faceswap/data/faces/fauci/*'


strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0", "/gpu:1"])
num_gpu = 2

# strategy = tf.distribute.MirroredStrategy(
#     devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
# )
# num_gpu = 4

# strategy = tf.distribute.MirroredStrategy(
#     devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"]
# )
# num_gpu = 8

# strategy = tf.distribute.MirroredStrategy(
#     devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7",
#              "/gpu:8", "/gpu:9", "/gpu:10", "/gpu:11", "/gpu:12", "/gpu:13", "/gpu:14", "/gpu:15"]
# )
# num_gpu = 16


def parse_image(filename):
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [input_shape[0], input_shape[1]])
  return (image, image[:, :, 0:1]), (image, image[:, :, 0:1])


def show(image):
  plt.figure()
  plt.imshow(image)
  plt.axis('off')
  plt.show()


with strategy.scope():

  # Create Model
  trainer_name = 'dfl-h128'
  model_dir = get_folder('/home/ubuntu/faceswap/trump_fauci_model_realface')
  gpus = 1
  configfile = None
  snapshot_interval = 25000
  no_logs = False
  warp_to_landmarks = False
  augment_color = False
  no_flip = True
  training_image_size = 256
  alignments_paths = None
  preview_scale = 50
  pingpong = False
  memory_saving_gradients = False
  optimizer_savings = False
  predict = False

  model = PluginLoader.get_model(trainer_name)(
      model_dir,
      gpus=gpus,
      configfile=configfile,
      snapshot_interval=snapshot_interval,
      no_logs=no_logs,
      warp_to_landmarks=warp_to_landmarks,
      augment_color=augment_color,
      no_flip=no_flip,
      training_image_size=training_image_size,
      alignments_paths=alignments_paths,
      preview_scale=preview_scale,
      pingpong=pingpong,
      memory_saving_gradients=memory_saving_gradients,
      optimizer_savings=optimizer_savings,
      predict=predict)


model_sources = [
    np.zeros(
        (batch_size * 64,
         input_shape[0],
         input_shape[1],
         input_shape[2])
    ),
    np.zeros(
        (batch_size * 64,
         input_shape[0],
         input_shape[1],
         1)
    )
]


model_targets = [
    np.zeros(
        (batch_size * 64,
         input_shape[0],
         input_shape[1],
         input_shape[2])
    ),
    np.zeros(
        (batch_size * 64,
         input_shape[0],
         input_shape[1],
         1)
    )
]

images_ds_a = tf.data.Dataset.list_files(
        dir_a).map(
        parse_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batch_size * num_gpu,
        drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

# Really fast
start_time = time.perf_counter()

# This works
model.predictors['a'].fit(
  model_sources, model_targets, 
  batch_size=4,
  epochs=2)

# This also works
# model.predictors['a'].fit(images_ds_a, epochs=2)

print("fit Execution time: {}".format(time.perf_counter() - start_time))
