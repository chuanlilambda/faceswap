import itertools
import random
import numpy as np
import cv2

import tensorflow as tf


from lib.utils import get_image_paths
from lib.training_data import TrainingDataGenerator
from plugins.train.trainer._base import TrainingAlignments

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


def read_image(list_name, idx):
	image = tf.io.read_file(list_name[idx])
	image = tf.image.decode_jpeg(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, [input_shape[0], input_shape[1]])

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


def gen():

	for i in itertools.count(1):
		idx_a = random.randrange(size_a)
		idx_b = random.randrange(size_b)

		image_a = read_image(list_images['a'], idx_a)
		image_b = read_image(list_images['b'], idx_b)

		mask_a = masks['a'][list_images['a'][idx_a]].mask
		mask_b = masks['b'][list_images['b'][idx_b]].mask

		mask_a = tf.image.resize(mask_a, [input_shape[0], input_shape[1]])
		mask_b = tf.image.resize(mask_b, [input_shape[0], input_shape[1]])

		yield (image_a, mask_a, image_b, mask_b)


dataset = tf.data.Dataset.from_generator(
	gen,
	(tf.float32, tf.float32, tf.float32, tf.float32),
	(
		tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
	 	tf.TensorShape([input_shape[0], input_shape[1], 1]),
	 	tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
	 	tf.TensorShape([input_shape[0], input_shape[1], 1])
	)
)


dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(
	batch_size,
	drop_remainder=True
)


for x in dataset.take(10):
	print(x[0].shape)
	print(x[1].shape)
	print(x[2].shape)
	print(x[3].shape)