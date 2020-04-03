"""
Define functions used in `define_task_sets_[type_task_set].py` for
[type_task_set] in {difficulty, size, similarity}.
"""

import numpy as np
import pandas as pd
from ..utils.metadata import (
    acc_vgg, distances_represent, mean_acc, mean_dist, std_acc, std_dist)

def mean_distance(inds):
    if len(inds) == 1:
        return 0
    dist = []
    for i in inds:
        dist.extend(distances_represent[i, np.setdiff1d(inds, i)])
    return np.mean(dist)

def mean_vgg_accuracy(inds):
    return np.mean(acc_vgg['accuracy'][inds])

def check_coverage(scores, interval_ends):
    intervals = [
        [interval_ends[i], interval_ends[i+1]]
        for i in range(len(interval_ends) - 1)]
    return np.all([np.any((l < scores) & (scores < u)) for l, u in intervals])

def check_dist_in_bounds(inds):
    dist = mean_distance(inds)
    return ((mean_dist - std_dist) < dist) & (dist < (mean_dist + std_dist))

def sample_below_acc(acc, task_set_size=50):
    return np.random.choice(
        acc_vgg.loc[acc_vgg['accuracy'] <= acc].index,
        size=task_set_size,
        replace=False)

def score_acc(inds):
    return (mean_vgg_accuracy(inds) - mean_acc) / std_acc

def score_dist(inds):
    return (mean_distance(inds) - mean_dist) / std_dist
