"""
Train a baseline attention network on examples from all ImageNet categories.
"""

gpu = input('GPU: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from ..utils.layers import ElementwiseAttention
from ..utils.models import attention_network
from ..utils.paths import path_imagenet, path_training, path_weights
from ..utils.training import model_train

model = attention_network(ElementwiseAttention(trainable=True))
model, history = model_train(model, 'dir', path_imagenet/'train/')
pd.DataFrame(history.history).to_csv(path_training/'baseline_attn_training.csv')
np.save(
    path_weights/'baseline_attn_weights.npy',
    model.get_layer('attention').get_weights()[0],
    allow_pickle=False)
