'''
>>> generator0 = datagen0.flow_from_directory(directory=path_data, **params_generator)
Found 1281167 images belonging to 1000 classes.

>>> DataGenerator(filepaths, path2label).n
1281167

>>> buildtime0, buildtime1, epochtime0
(2.2791130542755127, 1.316725492477417, 2752.835251569748)
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import numpy as np
import pandas as pd
from models import build_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from models_vgg2 import build_vgg2
from generator import DataGenerator
import time

data_partition = 'train'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_activations = '/home/freddie/activations-conv-split/'

params_generator = dict(
    class_mode='categorical', target_size=(224, 224), batch_size=256, shuffle=True)

params_testing = dict(
    use_multiprocessing=True, workers=7, verbose=True)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator0 = datagen.flow_from_directory(directory=path_data, **params_generator)
filepaths = path_activations + pd.Series(generator0.filenames).str.replace('.JPEG', '_conv5.npy')
path2label = {filepath:label for filepath, label in zip(filepaths, generator0.classes)}
generator1 = DataGenerator(filepaths, path2label)

time0 = time.time()
model0 = build_model()
buildtime0 = time.time()-time0

time0 = time.time()
model1 = build_vgg2()
buildtime1 = time.time()-time0

# time0 = time.time()
# scores0 = model0.evaluate_generator(
#     generator=generator0, steps=generator0.__len__(), **params_testing)
# epochtime0 = time.time()-time0
epochtime0 = 2752.835251569748

time0 = time.time()
scores1 = model1.evaluate_generator(
    generator=generator1, steps=generator1.__len__(), **params_testing)
epochtime1 = time.time()-time0
print('finished')

print(f'''
OLD MODEL
build time = {int(buildtime0)} seconds
epoch time = {int(epochtime0)} seconds
scores = {scores0}
{model0.metrics_names}

NEW MODEL
build time = {int(buildtime1)} seconds
epoch time = {int(epochtime1)} seconds
scores = {scores1}
{model1.metrics_names}
''')
