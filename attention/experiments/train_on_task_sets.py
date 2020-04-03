"""
For each task set, train an attention network on examples solely from the task
set.
"""

gpu = input('GPU: ')
type_task_set = input('Task-set type in {diff, size, sim}: ')
version_wnids = input('Version number (WNIDs): ')
version_weights = input('Version number (training/weights): ')
start = int(input('Start task set: '))
stop = int(input('Stop task set (inclusive): '))

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from ..utils.layers import ElementwiseAttention
from ..utils.models import attention_network
from ..utils.paths import (path_dataframes, path_imagenet, path_init_model,
    path_training, path_weights)
from ..utils.training import model_train

model = attention_network(ElementwiseAttention(trainable=True))
model.save_weights(path_init_model)

for i in range(start, stop+1):
    id_wnids = f'{type_task_set}_v{version_wnids}_{i:02}'
    id_weights = f'{type_task_set}_v{version_weights}_{i:02}'
    print(f'\nTraining on {id_wnids}')
    model.load_weights(path_init_model)
    args_train = [
        pd.read_csv(path_dataframes/f'{id_wnids}_df.csv'),
        path_imagenet/'train/']
    model, history = model_train(model, 'df', *args_train)
    pd.DataFrame(history.history).to_csv(
        path_training/f'{id_weights}_training.csv')
    np.save(
        path_weights/f'{id_weights}_weights.npy',
        model.get_layer('attention').get_weights()[0],
        allow_pickle=False)
