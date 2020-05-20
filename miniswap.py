import logging
import numpy as np


from plugins.plugin_loader import PluginLoader
from lib.utils import get_folder

import tensorflow as tf


# global variables
batch_size = 16
input_shape = (128, 128, 3)
output_shape = (128, 128, 3)
trainer_name = 'dfl-h128'

# Synthetic Data
model_inputs = [
    np.zeros(
        (batch_size,
         input_shape[0],
         input_shape[1],
         input_shape[2])
    ),
    np.zeros(
        (batch_size,
         input_shape[0],
         input_shape[1],
         1)
    )
]
    
model_targets = [
    np.zeros(
        (batch_size,
         input_shape[0],
         input_shape[1],
         input_shape[2])
    ),
    np.zeros(
        (batch_size,
         input_shape[0],
         input_shape[1],
         1)
    )
]


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
alignments_paths = {'a': '/home/ubuntu/faceswap/data/src/trump/trump_alignments.fsa', 'b': '/home/ubuntu/faceswap/data/src/fauci/fauci_alignments.fsa'}
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