'''
Check that an attention model with attention weights set to 1 achieves the same
performance as a VGG16 without an attention layer.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from layers import FixedWeightAttention
from models import build_model
from testing import evaluate_model

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/'
W = np.ones((1, 7, 7, 512))
model = build_model(FixedWeightAttention(W), train=False)
scores = evaluate_model(model, path_data)
print(f'{model.metrics_names} = {scores}')
