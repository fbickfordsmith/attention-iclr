"""
Test attention networks on ImageNet.
"""

gpu = input('GPU: ')
type_task_set = input('Task-set type in {diff, size, sim}: ')
version_wnids = input('Version number (WNIDs): ')
version_weights = input('Version number (weights): ')
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
from ..utils.paths import (path_imagenet, path_init_model, path_results,
    path_task_sets, path_weights)
from ..utils.testing import model_predict, predictions_metrics

model = attention_network(ElementwiseAttention(trainable=False))
model.save_weights(path_init_model)
path_task_sets = (
    path_task_sets/f'{type_task_set}_v{version_wnids}_wnids.csv')
task_sets = [row for row in csv.reader(open(path_task_sets), delimiter=',')]
scores_in, scores_out = [], []

for i in range(start, stop+1):
    id_weights = f'{type_task_set}_v{version_weights}_{i:02}'
    print(f'\nTesting on {id_weights}')
    weights = np.load(path_weights/f'{id_weights}_weights.npy')
    model.load_weights(path_init_model)
    model.get_layer('attention').set_weights([weights])
    predictions, generator = model_predict(
        model, 'dir', path_imagenet/'val_white/')
    wnid2ind = generator.class_indices
    labels = generator.classes
    inds_in = []
    for wnid in task_sets[i]:
        inds_in.extend(np.flatnonzero(labels == wnid2ind[wnid]))
    inds_out = np.setdiff1d(range(generator.n), inds_in)
    print(
        f'{len(inds_in)} in-set examples, {len(inds_out)} out-of-set examples')
    scores_in.append(predictions_metrics(predictions, labels, inds_in))
    scores_out.append(predictions_metrics(predictions, labels, inds_out))

cols_array = ['loss_in', 'acc_top1_in', 'acc_top5_in', 'loss_out',
    'acc_top1_out', 'acc_top5_out']
cols_save = ['loss_in', 'loss_out', 'acc_top1_in', 'acc_top1_out',
    'acc_top5_in', 'acc_top5_out']

scores_all = np.concatenate((np.array(scores_in), np.array(scores_out)), axis=1)
scores_df = pd.DataFrame(scores_all, columns=cols_array)
scores_df[cols_save].to_csv(
    (path_results/
    f'{type_task_set}_v{version_weights}_{start:02}-{stop:02}_results.csv'))
