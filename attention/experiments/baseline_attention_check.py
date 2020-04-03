"""
Sanity check: test an attention network with attention weights set to 1.
Agreement with the results produced by `vgg16_testing.py` implies that the
attention network works as expected.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from ..utils.layers import ElementwiseAttention
from ..utils.models import attention_network
from ..utils.paths import path_imagenet, path_results
from ..utils.testing import predictions_accuracy_classwise, model_predict

model = attention_network(ElementwiseAttention(trainable=True))
predictions, generator = model_predict(
    model, 'dir', path_imagenet/data_partition)
accuracy = predictions_accuracy_classwise(predictions, generator)
accuracy.to_csv(path_results/'untrained_attn_results.csv', index=False)
