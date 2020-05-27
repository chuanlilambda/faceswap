import itertools
import random

import tensorflow as tf


from lib.utils import get_image_paths

path_a = '/home/ubuntu/faceswap/data/faces/trump'
path_b = '/home/ubuntu/faceswap/data/faces/fauci'
input_shape = (128, 128, 3)

def get_images(path_a, path_b):
    """ Check the image folders exist and contains images and obtain image paths.
    Returns
    -------
    list_a: The image paths for side a. 
    list_b: The image paths for side b.
    """

    # return images
    list_a = get_image_paths(path_a)
    list_b = get_image_paths(path_b)
    return list_a, list_b

list_a, list_b = get_images(path_a, path_b)
size_a = len(list_a)
size_b = len(list_b)


def read_image(list_name, idx):
	image = tf.io.read_file(list_name[idx])
	image = tf.image.decode_jpeg(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, [input_shape[0], input_shape[1]])

	return image


def gen():

	for i in itertools.count(1):
		idx_a = random.randrange(size_a)
		idx_b = random.randrange(size_b)

		image_a = read_image(list_a, idx_a)
		image_b = read_image(list_b, idx_b)

		yield (image_a, image_b)


dataset = tf.data.Dataset.from_generator(
	gen,
	(tf.float32, tf.float32),
	(tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]),
	 tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]]))
)


for x in dataset.take(10):
	print(x[0].shape)
	print(x[1].shape)