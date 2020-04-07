"""
For each ImageNet category, take the VGG16 representations of images
belonging to it, and compute the mean of these. The VGG16 representation of an
image is found by computing a forward pass through the network and looking at
the activation of the penultimate layer.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import tensorflow as tf
from ..utils.metadata import wnids
from ..utils.paths import path_imagenet, path_representations
from ..utils.testing import model_predict

vgg = tf.keras.applications.VGG16()
model = tf.keras.models.Sequential()
for layer in vgg.layers[:-1]:
    model.add(layer)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
generator = datagen.flow_from_directory(directory=path_imagenet/'train/')
df = pd.DataFrame({
    'filename':generator.filenames,
    'wnid':pd.Series(generator.filenames).str.split('/', expand=True)[0]})
mean_representations = []

for i, wnid in enumerate(wnids):
    generator_wnid = datagen.flow_from_dataframe(
        dataframe=df.loc[df['wnid']==wnid],
        directory=path_imagenet/'train/',
        target_size=(224,224),
        batch_size=256,
        shuffle=False,
        class_mode=None)
    representations_wnid = generator_wnid.predict(
        x=generator_wnid,
        steps=len(generator_wnid),
        use_multiprocessing=False,
        verbose=True)
    mean_representations.append(np.mean(representations_wnid, axis=0))

np.save(
    path_representations,
    np.array(mean_representations),
    allow_pickle=False)
