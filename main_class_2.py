import itertools
import random
import numpy as np
import cv2
import time

import tensorflow as tf


from lib.utils import get_folder, get_image_paths
from lib.training_data import TrainingDataGenerator
from plugins.train.trainer._base import TrainingAlignments
from plugins.plugin_loader import PluginLoader

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


image_path_a = '/home/ubuntu/faceswap/data/faces/trump'
image_path_b = '/home/ubuntu/faceswap/data/faces/fauci'
alignment_path_a = '/home/ubuntu/faceswap/data/src/trump/trump_alignments.fsa'
alignment_path_b = '/home/ubuntu/faceswap/data/src/fauci/fauci_alignments.fsa'
input_shape = (128, 128, 3)
mask_type = 'unet-dfl'
mask_blur_kernel = 3
mask_threshold = 4
learn_mask = True
batch_size = 4
num_samples = 1024

# strategy = tf.distribute.MirroredStrategy(
#     devices=["/gpu:0", "/gpu:1"])
# num_gpu = 2


def get_images(path_a, path_b):
    """ Check the image folders exist and contains images and obtain image paths.

    Returns
    -------
    dict
        The image paths for each side. The key is the side, the value is the list of paths
        for that side.
    """
    images = dict()
    for side in ("a", "b"):
        # image_dir = getattr(self._args, "input_{}".format(side))
        if side == "a":
            image_dir = path_a
        elif side == "b":
            image_dir = path_b

        images[side] = get_image_paths(image_dir)

    return images


def alignments_paths(path_a, path_b):
    """ 
    Returns
    -------
    dict: The alignments paths for each of the source and destination faces. Key is the
        side, value is the path to the alignments file 
    """
    alignments_paths = dict()
    for side in ("a", "b"):
        if side == "a":
            alignments_path = path_a
        elif side == "b":
            alignments_path = path_b
        alignments_paths[side] = alignments_path

    return alignments_paths


# def read_image(list_name, idx):
#     image = tf.io.read_file(list_name[idx])
#     image = tf.image.decode_jpeg(image)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, [input_shape[0], input_shape[1]])

#     return image


def read_image(list_name, idx):
    image = cv2.imread(list_name[idx])
    image = cv2.resize(image,
                       (input_shape[0], input_shape[1]),
                       cv2.INTER_AREA) / 255.
    return image

list_images = get_images(image_path_a, image_path_b)
alignments_paths = alignments_paths(alignment_path_a, alignment_path_b)

training_opts = {
    "alignments": alignments_paths,
    "mask_type": mask_type,
    "mask_blur_kernel": mask_blur_kernel,
    "mask_threshold": mask_threshold,
    "learn_mask": learn_mask}

size_a = len(list_images['a'])
size_b = len(list_images['b'])

alignments = TrainingAlignments(training_opts, list_images)
landmarks = alignments.masks
masks = alignments.masks

# ---------------------------------------------------------------------------
# class ArtificialDataset(tf.data.Dataset):
#     def _generator():
#         for i in range(num_samples):
#             idx_a = random.randrange(size_a)
#             idx_b = random.randrange(size_b)

#             image_a = read_image(list_images['a'], idx_a)
#             image_b = read_image(list_images['b'], idx_b)

#             mask_a = masks['a'][list_images['a'][idx_a]].mask
#             mask_b = masks['b'][list_images['b'][idx_b]].mask

#             mask_a = tf.image.resize(mask_a, [input_shape[0], input_shape[1]])
#             mask_b = tf.image.resize(mask_b, [input_shape[0], input_shape[1]])

#             yield (image_a, mask_a), (image_a, mask_a)
        
#     def __new__(cls, num_samples=4):
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             ((tf.float32, tf.float32), (tf.float32, tf.float32)),
#             (
#                 (tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
#                 tf.TensorShape([input_shape[0], input_shape[1], 1])),
#                 (tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
#                 tf.TensorShape([input_shape[0], input_shape[1], 1]))
#             )
#         )

# Change parameter of range does impact speed
# dataset = tf.data.Dataset.range(2).interleave(
#     ArtificialDataset,
#     num_parallel_calls=tf.data.experimental.AUTOTUNE
# ).batch(
#     batch_size,
#     drop_remainder=True
# ).prefetch(
#     tf.data.experimental.AUTOTUNE
# )

# ---------------------------------------------------------------------------
def gen():

    im = np.zeros(
            (input_shape[0],
             input_shape[1],
             input_shape[2])
        )

    mask = np.zeros(
            (input_shape[0],
             input_shape[1],
             1)
        )

    for i in itertools.count(1):
        idx_a = random.randrange(size_a)
        idx_b = random.randrange(size_b)

        image_a = read_image(list_images['a'], idx_a)
        image_b = read_image(list_images['b'], idx_b)

        mask_a = masks['a'][list_images['a'][idx_a]].mask
        mask_b = masks['b'][list_images['b'][idx_b]].mask

        # mask_a = tf.image.resize(mask_a, [input_shape[0], input_shape[1]])
        # mask_b = tf.image.resize(mask_b, [input_shape[0], input_shape[1]])

        mask_size = mask_a.shape[1]
        interpolator = cv2.INTER_CUBIC if mask_size < input_shape[1] else cv2.INTER_AREA
        mask_a = cv2.resize(mask_a,
                       (input_shape[0], input_shape[1]),
                       interpolator) / 255.
        mask_a = np.expand_dims(mask_a, -1)

        mask_size = mask_b.shape[1]
        interpolator = cv2.INTER_CUBIC if mask_size < input_shape[1] else cv2.INTER_AREA
        mask_b = cv2.resize(mask_b,
                       (input_shape[0], input_shape[1]),
                       interpolator) / 255.
        mask_b = np.expand_dims(mask_b, -1)

        # yield image_a, image_b
        yield (image_a, mask_a), (image_b, mask_b)
        # yield (image_a, mask), (image_b, mask)
        # yield (im, mask_a), (im, mask_b)

dataset = tf.data.Dataset.from_generator(
    gen,
    ((tf.float32, tf.float32), (tf.float32, tf.float32)),
    (
        (tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
        tf.TensorShape([input_shape[0], input_shape[1], 1])),
        (tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
        tf.TensorShape([input_shape[0], input_shape[1], 1]))
    )
)

# dataset = tf.data.Dataset.from_generator(
#     gen,
#     (tf.float32, tf.float32),
#     (tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
#     tf.TensorShape([input_shape[0], input_shape[1], 1]))
# )

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(
    batch_size,
    drop_remainder=True
)

# with strategy.scope():
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


start_time = time.perf_counter()
model.predictors['a'].fit(dataset, epochs=2, steps_per_epoch=20)
print("fit Execution time: {}".format(time.perf_counter() - start_time))


# # t_start = time.time()
# count = 0 
# for x in dataset.take(1):
#     print(count)
#     count += 1
#     # print(len(x))
#     # print(type(x[0]))
#     print(np.max(x[0][0]))
#     print(np.max(x[0][1]))   
#     # print(x[0].shape)
#     # print(x[1].shape)
# # t_end = time.time()
# # print(t_end - t_start)


# import sys
# sys.exit()

# Syn DATA ---------------------------

# model_sources = [
#     np.zeros(
#         (batch_size * 64,
#          input_shape[0],
#          input_shape[1],
#          input_shape[2])
#     ),
#     np.zeros(
#         (batch_size * 64,
#          input_shape[0],
#          input_shape[1],
#          1)
#     )
# ]


# model_targets = [
#     np.zeros(
#         (batch_size * 64,
#          input_shape[0],
#          input_shape[1],
#          input_shape[2])
#     ),
#     np.zeros(
#         (batch_size * 64,
#          input_shape[0],
#          input_shape[1],
#          1)
#     )
# ]

# model.predictors['a'].fit(
#   model_sources, model_targets, 
#   batch_size=4,
#   epochs=2)
