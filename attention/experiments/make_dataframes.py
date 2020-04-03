"""
For each task set, make a dataframe containing filepaths, labels and filenames
for all the examples in the task set.
"""

gpu = input('GPU: ')
type_task_set = input('Task-set type in {diff, size, sim}: ')
version_wnids = input('Version number (WNIDs): ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from ..utils.paths import path_dataframes, path_imagenet, path_task_sets

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
generator = datagen.flow_from_directory(directory=path_imagenet/'train/')
df = pd.DataFrame({
    'filename':generator.filenames,
    'class':pd.Series(generator.filenames).str.split('/', expand=True)[0]})

with open(path_task_sets/f'{type_task_set}_v{version_wnids}_wnids.csv') as f:
    task_sets = [row for row in csv.reader(f, delimiter=',')]

for i, task_set in enumerate(task_sets):
    inds_in_set = []
    for wnid in task_set:
        inds_in_set.extend(np.flatnonzero(df['class']==wnid))
    df.iloc[inds_in_set].to_csv(
        path_dataframes/f'{type_task_set}_v{version_wnids}_{i:02}_df.csv',
        index=False)
