import logging
import numpy as np
import matplotlib.pyplot as plt
import time

from plugins.plugin_loader import PluginLoader
from lib.utils import get_folder, get_image_paths

import tensorflow as tf


# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

# global variables
num_gpu = 1
batch_size = 128
input_shape = (128, 128, 3)
output_shape = (128, 128, 3)
trainer_name = 'dfl-h128'
dir_a = '/home/ubuntu/faceswap/data/faces/trump/*'
dir_b = '/home/ubuntu/faceswap/data/faces/fauci/*'


strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0", "/gpu:1"])
num_gpu = 2

# strategy = tf.distribute.MirroredStrategy(
#     devices=["/gpu:0"])
# num_gpu = 1

# strategy = tf.distribute.MirroredStrategy(
#     devices=["/gpu:1"])
# num_gpu = 1


## Synthetic Data with mask
# model_inputs = [
#     np.zeros(
#         (batch_size,
#          input_shape[0],
#          input_shape[1],
#          input_shape[2])
#     ),
#     np.zeros(
#         (batch_size,
#          input_shape[0],
#          input_shape[1],
#          1)
#     )
# ]

## Synthetic Data without mask
# model_targets = [
#     np.zeros(
#         (batch_size,
#          input_shape[0],
#          input_shape[1],
#          input_shape[2])
#     ),
#     np.zeros(
#         (batch_size,
#          input_shape[0],
#          input_shape[1],
#          1)
#     )
# ]

# model_inputs = [
#     np.zeros(
#         (batch_size,
#          input_shape[0],
#          input_shape[1],
#          input_shape[2])
#     )
# ]
    
# model_targets = [
#     np.zeros(
#         (batch_size,
#          input_shape[0],
#          input_shape[1],
#          input_shape[2])
#     )
# ]

## Real data without mask
# def get_images():
#     """ Check the image folders exist and contains images and obtain image paths.

#     Returns
#     -------
#     dict
#         The image paths for each side. The key is the side, the value is the list of paths
#         for that side.
#     """
#     images = dict()
#     for side in ("a", "b"):
#         # image_dir = getattr(self._args, "input_{}".format(side))
#         if side == "a":
#             image_dir = '/home/ubuntu/faceswap/data/faces/trump'
#         if side == "b":
#             image_dir = '/home/ubuntu/faceswap/data/faces/fauci'

#         images[side] = get_image_paths(image_dir)

#     return images

# images = get_images()

# print(images)


def parse_image(filename):
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [input_shape[0], input_shape[1]])
  return image, image

def show(image):
  plt.figure()
  plt.imshow(image)
  plt.axis('off')
  plt.show()

# list_ds_a = tf.data.Dataset.list_files(
#         [dir_a, dir_b])

# for f in list_ds_a.take(5):
#     print(f)


# import sys
# sys.exit()

with strategy.scope():

# # Original: 34.81 sec
# images_ds_a = tf.data.Dataset.list_files(
#         dir_a).map(
#         parse_image).batch(
#         batch_size,
#         drop_remainder=True)

    images_ds_a = tf.data.Dataset.list_files(
            dir_a).map(
            parse_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            batch_size * num_gpu,
            drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)


# for image_in, image_out in images_ds_a.take(2):
#     print(type(image_in))
#     print(type(image_out))
#     # show(image_in)


# Interleave


# # num_parallel_calls: 35.42 sec
# images_ds_a = tf.data.Dataset.list_files(
#         dir_a).map(
#         parse_image,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
#         batch_size,
#         drop_remainder=True)

# Prefetch: 35.08 sec
# images_ds_a = tf.data.Dataset.list_files(
#         dir_a).map(
#         parse_image,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
#         batch_size,
#         drop_remainder=True).prefetch(
#         tf.data.experimental.AUTOTUNE)


    # images_ds_b = tf.data.Dataset.list_files(
    #         dir_b).map(
    #         parse_image,
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
    #         batch_size,
    #         drop_remainder=True).prefetch(
    #         tf.data.experimental.AUTOTUNE)

    # for image_in, image_out in images_ds_a.take(2):
    #     print(type(image_in))
    #     print(type(image_out))
    #     # show(image_in)


    # for f in list_ds_a.take(5):
    #   image = parse_image(f)
    #   print(type(image))
    #   show(image)

    # for f in list_ds_b.take(5):
    #   image = parse_image(f)
    #   print(type(image))
    #   show(image)

# import sys
# sys.exit()

    # Model

    # Create Model
    trainer_name = 'dfl-h128'
    model_dir = get_folder('/home/ubuntu/faceswap/trump_fauci_model_realface')
    gpus = 2
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


# # Warm up
# for data_a in images_ds_a.take(1):
#     model.predictors['a'].train_on_batch(data_a[0], data_a[1])

# start_time = time.perf_counter()
# num_iter = 40
# for data_a in images_ds_a.take(num_iter):
#     model.predictors['a'].train_on_batch(data_a[0], data_a[1])
# tf.print("train_on_batch Execution time:", time.perf_counter() - start_time)



# Really fast
start_time = time.perf_counter()
model.predictors['a'].fit(images_ds_a, epochs=2)
print("fit Execution time: {}".format(time.perf_counter() - start_time))

# Slow because data and training are not pipelined
# num_iter = 100
# for data_a in images_ds_a.take(num_iter):
#     model.predictors['a'].train_on_batch(data_a[0], data_a[1])
#     model.predictors['b'].train_on_batch(data_a[0], data_a[1])

# Slow because data and training are not pipelined
# num_iter = 100
# for data_a, data_b in zip(images_ds_a.take(num_iter), images_ds_b.take(num_iter)):
#     print('h')
#     model.predictors['a'].train_on_batch(data_a[0], data_a[1])
#     model.predictors['b'].train_on_batch(data_b[0], data_b[1])

    # show(image_in)

# for i in range(100):
    # model.predictors['a'].train_on_batch(model_inputs, model_targets)
#     print(i)
    # model.predictors['a'].train_on_batch(model_inputs, model_targets)
#     model.predictors['b'].train_on_batch(model_inputs, model_targets)
