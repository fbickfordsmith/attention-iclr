"""
Define metadata variables used throughout the repository.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from ..utils.paths import path_metadata, path_representations, path_results

wnids = np.loadtxt(path_metadata/'imagenet_class_wnids.txt', dtype=str)
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

acc_baseline = pd.read_csv(path_results/'baseline_attn_results.csv')
acc_vgg = pd.read_csv(path_results/'vgg16_results.csv')
mean_acc = np.mean(acc_vgg['accuracy'])
std_acc = np.std(acc_vgg['accuracy'])

distances_represent = cosine_distances(np.load(path_representations))
mean_dist = np.mean(squareform(distances_represent, checks=False))
std_dist = np.std(squareform(distances_represent, checks=False))
