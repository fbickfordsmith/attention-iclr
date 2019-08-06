
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
path_activations = '/home/freddie/activations-conv/'
path_split = '/home/freddie/activations-conv-split/'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'

wnids = [line.rstrip('\n') for line in open(path_synsets)]
generator = ImageDataGenerator().flow_from_directory(directory=path_data)
class_filename = pd.Series(generator.filenames).str.split('/', expand=True)
df = pd.DataFrame()
df['filename'] = class_filename[1].str.split('.', expand=True)[0]
df['class'] = class_filename[0]

for i, wnid in enumerate(wnids):
    path_class = path_split + wnid
    os.makedirs(path_class)
    filenames = (df.loc[df['class']==wnid])['filename']
    activations = np.load(f'{path_activations}class{i:04}_activations_conv.npy')
    for f, a in zip(filenames, activations):
        np.save(f'{path_class}/{f}_conv5.npy', a, allow_pickle=False)
