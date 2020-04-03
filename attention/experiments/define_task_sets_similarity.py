"""
Define a set of 20 similarity-based task sets. These are subsets of ImageNet
categories that we choose to have varying perceptual similarity (mean pairwise
cosine similarity of VGG16 representations) but equal size and approx equal
difficulty.

Method:
1. Sample 5 seeds.
2. For each seed,
    a. For k in {50, 366, 682, 999},
        i.  Uniformly sample 49 indices from the seed's k nearest neighbours.
        ii. Compute the cosine distance (= 1 - cosine similarity) and accuracy
            of the sampled task set.
3. Check that the sampled task sets give good coverage of similarity values
    between 0.1 and 0.6.
4. Keep the sampled task sets if their accuracy score (normalised distance from
    mean VGG16 accuracy) is better than any previous score.
"""

version_wnids = input('Version number (WNIDs): ')

import numpy as np
import pandas as pd
from ..utils.metadata import ind2wnid, distances_represent
from ..utils.paths import path_task_sets
from ..utils.task_sets import (check_coverage, mean_distance, mean_vgg_accuracy,
    score_acc)

num_seeds = 5
task_set_size = 50
inds_end = np.linspace(50, 999, 4, dtype=int)
interval_ends = np.arange(0.1, 0.65, 0.05)
intervals_covered = False
acc_bestscore = np.inf
inds_best = None

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled, dist, acc = [], [], []
    inds_seed = np.random.choice(1000, size=num_seeds, replace=False)
    for ind_seed in inds_seed:
        inds_sorted = np.argsort(distances_represent[ind_seed])[1:] # 1 => don't include seed index
        for ind_end in inds_end:
            inds_task_set = np.random.choice(
                inds_sorted[:ind_end], size=task_set_size-1, replace=False) # 'sampled nearest neighbour'
            inds_task_set = np.insert(inds_task_set, 0, ind_seed)
            inds_sampled.append(inds_task_set)
            dist.append(mean_distance(inds_task_set))
            acc.append(score_acc(inds_task_set))
    intervals_covered = check_coverage(np.array(dist), interval_ends)
    acc_score = np.max(np.abs(acc))
    if intervals_covered and (acc_score < acc_bestscore):
        inds_best = inds_sampled
        acc_bestscore = acc_score

if inds_best is not None:
    print(
        'Accuracy:', [round(mean_vgg_accuracy(inds), 2) for inds in inds_best],
        '\nDistance:', [round(mean_distance(inds), 2) for inds in inds_best])
    wnids_best = np.vectorize(ind2wnid.get)(inds_best)
    pd.DataFrame(wnids_best).to_csv(
        path_task_sets/f'sim_v{version_wnids}_wnids.csv',
        header=False,
        index=False)
else:
    print('Suitable task sets not found')
