"""
Define functions used for processing results.
"""

import csv
import os
import numpy as np
import pandas as pd
from ..utils.metadata import acc_baseline, acc_vgg, wnid2ind
from ..utils.paths import path_task_sets, path_results, path_training
from ..utils.task_sets import mean_distance

def load_task_sets(type_task_set, version_wnids):
    id_wnids = f'{type_task_set}_v{version_wnids}'
    with open(path_task_sets/f'{id_wnids}_wnids.csv') as f:
        return [row for row in csv.reader(f, delimiter=',')]

def task_set_difficulty(task_sets):
    difficulty = []
    for ts in task_sets:
        inds_in = [wnid2ind[wnid] for wnid in ts]
        difficulty.append(1 - np.mean(acc_vgg['accuracy'][inds_in]))
    return pd.Series(difficulty, name='difficulty')

def task_set_size(task_sets):
    return [len(ts) for ts in task_sets]

def task_set_similarity(task_sets):
    similarity = []
    for ts in task_sets:
        inds_in = [wnid2ind[wnid] for wnid in ts]
        similarity.append(1 - mean_distance(inds_in))
    return pd.Series(similarity, name='similarity')

def task_set_baseline_accuracy(task_sets):
    acc_base = []
    for ts in task_sets:
        inds_in = [wnid2ind[wnid] for wnid in ts]
        inds_out = np.setdiff1d(range(1000), inds_in)
        acc_base.append([
            np.mean(acc_baseline['accuracy'][inds_in]),
            np.mean(acc_baseline['accuracy'][inds_out])])
    return pd.DataFrame(acc_base, columns=('acc_base_in', 'acc_base_out'))

def task_set_epochs(id_weights):
    return [
        len(pd.read_csv(path_training/filename, index_col=0))
        for filename in sorted(os.listdir(path_training))
        if id_weights in filename]

def task_set_summary(type_task_set, version_wnids, version_weights):
    id_weights = f'{type_task_set}_v{version_weights}'
    task_sets = load_task_sets(type_task_set, version_wnids)
    df0 = task_set_baseline_accuracy(task_sets)
    df1 = pd.read_csv(path_results/f'{id_weights}_results.csv', index_col=0)
    return pd.DataFrame({
        'difficulty':task_set_difficulty(task_sets),
        'size':task_set_size(task_sets),
        'similarity':task_set_similarity(task_sets),
        'acc_change_in':(df1['acc_top1_in']-df0['acc_base_in']),
        'acc_change_out':(df1['acc_top1_out']-df0['acc_base_out']),
        'num_epochs':task_set_epochs(id_weights)})
