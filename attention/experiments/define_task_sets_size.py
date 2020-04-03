"""
Define a set of 11 size-based task sets. These are subsets of ImageNet
categories that we choose to have varying size (number of categories) but approx
equal difficulty and perceptual similarity.

Method:
1. For 10,000 repeats
    a. Sample a candidate set of task sets, {C_1', ..., C_11'}.
    b. Compute acc = [normalised_deviation_from_mean_acc(C_i) for i = 1:11].
    c. Compute dist = [normalised_deviation_from_mean_dist(C_i) for i = 1:11].
    d. If std(concat(acc, dist)) < current_lowest_std, keep task set.
"""

version_wnids = input('Version number (WNIDs): ')

import numpy as np
import pandas as pd
from ..utils.metadata import ind2wnid
from ..utils.paths import path_task_sets
from ..utils.task_sets import (mean_distance, mean_vgg_accuracy, score_acc,
    score_dist)

task_set_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
accdist_bestscore = np.inf

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled = [
        np.random.choice(1000, size=s, replace=False) for s in task_set_sizes]
    accdist = [score_acc(inds) for inds in inds_sampled]
    accdist.extend([score_dist(inds) for inds in inds_sampled[1:]]) # first task set always has distance of 0
    accdist_score = np.max(np.abs(accdist))
    if accdist_score < accdist_bestscore:
        inds_best = inds_sampled
        accdist_bestscore = accdist_score

print(
    'Accuracy:', [round(mean_vgg_accuracy(inds), 2) for inds in inds_best],
    '\nDistance:', [round(mean_distance(inds), 2) for inds in inds_best])

with open(path_task_sets/f'size_v{version_wnids}_wnids.csv', 'w') as f:
    for inds_task_set in inds_best:
        csv.writer(f).writerow([ind2wnid[i] for i in inds_task_set])
