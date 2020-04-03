"""
Test a baseline attention network on ImageNet.
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
from ..utils.paths import path_imagenet, path_results, path_weights
from ..utils.testing import predictions_accuracy_classwise, model_predict

model = attention_network(ElementwiseAttention(trainable=False))
weights = np.load(path_weights/'baseline_attn_weights.npy')
model.get_layer('attention').set_weights([weights])
predictions, generator = model_predict(
    model, 'dir', path_imagenet/data_partition)
accuracy = predictions_accuracy_classwise(predictions, generator)
accuracy.to_csv(path_results/'baseline_attn_results.csv', index=False)
