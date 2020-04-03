"""
Define a set of 20 difficulty-based task sets. These are subsets of ImageNet
categories that we choose to have varying difficulty (mean error rate of VGG16)
but equal size and approx equal perceptual similarity.

Method:
1. Sort all 1000 ImageNet categories by difficulty (= 1 - VGG16 accuracy).
2. Split this sorted set into 20 disjoint, sorted sets of 50 categories.
3. Sample 5 additional sets in order to get better coverage of task-set
    accuracies in the range [0.2, 0.4].
"""

version_wnids = input('Version number (WNIDs): ')

import numpy as np
import pandas as pd
from ..utils.metadata import acc_vgg, ind2wnid
from ..utils.paths import path_task_sets
from ..utils.task_sets import (check_coverage, mean_distance, mean_vgg_accuracy,
    sample_below_acc, score_dist)

acc_vgg.sort_values(by='accuracy', ascending=True, inplace=True)
inds_split = np.array([list(inds) for inds in np.split(acc_vgg.index, 20)])
thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]
interval_ends = [0.2, 0.25, 0.3, 0.35, 0.4]
intervals_covered = False
dist_bestscore = np.inf
inds_best = None

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled = np.array([sample_below_acc(t) for t in thresholds])
    acc = [mean_vgg_accuracy(inds) for inds in inds_sampled]
    dist = [score_dist(inds) for inds in inds_sampled]
    intervals_covered = check_coverage(np.array(acc), interval_ends)
    dist_score = np.max(np.abs(dist)) # similar results with dist_score = np.std(dist)
    if intervals_covered and (dist_score < dist_bestscore):
        inds_best = inds_sampled
        dist_bestscore = dist_score

if inds_best is not None:
    print(
        'Accuracy:', [round(mean_vgg_accuracy(inds), 2) for inds in inds_best],
        '\nDistance:', [round(mean_distance(inds), 2) for inds in inds_best])
    inds_all = np.concatenate((inds_split, inds_best), axis=0)
    wnids_all = np.vectorize(ind2wnid.get)(inds_all)
    pd.DataFrame(wnids_all).to_csv(
        path_task_sets/f'diff_v{version_wnids}_wnids.csv',
        header=False,
        index=False)
else:
    print('Suitable task sets not found')
