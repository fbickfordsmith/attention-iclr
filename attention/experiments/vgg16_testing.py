"""
Test a pretrained VGG16 on ImageNet.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
import tensorflow as tf
from ..utils.paths import path_imagenet, path_results
from ..utils.testing import predictions_accuracy_classwise, model_predict

model = tf.keras.applications.vgg16.VGG16()
predictions, generator = model_predict(
    model, 'dir', path_imagenet/data_partition)
accuracy = predictions_accuracy_classwise(predictions, generator)
accuracy.to_csv(path_results/'vgg16_results.csv', index=False)
