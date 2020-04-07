"""
Define paths used throughout the repository.
"""

import pathlib
import sys

if sys.platform == 'linux':
    path_repo = pathlib.Path('/home/freddie/attention/')
    path_imagenet = pathlib.Path('/fast-data/datasets/ILSVRC/2012/clsloc/')
    path_init_model = pathlib.Path('/home/freddie/initialised_model.h5')
else:
    path_repo = pathlib.Path('/Users/fbickfordsmith/attention-iclr/')

path_task_sets = path_repo/'data/task_sets/'
path_metadata = path_repo/'data/metadata/'
path_representations = path_repo/'data/representations.npy'
path_results = path_repo/'data/results/'
path_training = path_repo/'data/training/'
path_weights = path_repo/'data/weights/'
