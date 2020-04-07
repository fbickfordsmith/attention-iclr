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

import csv
import numpy as np
import pandas as pd
from ..utils.layers import ElementwiseAttention
from ..utils.models import attention_network
from ..utils.paths import (path_dataframes, path_imagenet, path_init_model,
    path_training, path_weights)
from ..utils.training import (dataframe_imagenet_train, dataframe_task_set,
    model_train)

model = attention_network(ElementwiseAttention(trainable=True))
model.save_weights(path_init_model)
df_train_all = dataframe_imagenet_train()
with open(path_task_sets/f'{type_task_set}_v{version_wnids}_wnids.csv') as f:
    task_sets = [row for row in csv.reader(f, delimiter=',')]

for i in range(start, stop+1):
    id_wnids = f'{type_task_set}_v{version_wnids}_{i:02}'
    id_weights = f'{type_task_set}_v{version_weights}_{i:02}'
    print(f'\nTraining on {id_wnids}')
    model.load_weights(path_init_model)
    df_ts = dataframe_task_set(df_train_all, task_sets[i])
    model, history = model_train(model, 'df', df_ts, path_imagenet/'train/')
    pd.DataFrame(history.history).to_csv(
        path_training/f'{id_weights}_training.csv')
    np.save(
        path_weights/f'{id_weights}_weights.npy',
        model.get_layer('attention').get_weights()[0],
        allow_pickle=False)
