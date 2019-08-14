'''
For each image in the ImageNet training set, get the VGG16 representation at the
penultimate layer (ie, the activation of the layer just before the softmax).
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input
from testing import predict_model

data_partition = 'train'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_activations = '/home/freddie/activations/'

# copy all layers except for the final one
vgg = VGG16(weights='imagenet')
input = Input(batch_shape=(None, 224, 224, 3))
output = vgg.layers[1](input)
for layer in vgg.layers[2:-1]:
    output = layer(output)
model = Model(input, output)
activations, generator = predict_model(model, path_data)

# need to split this up to limit memory usage

for i in range(generator.num_classes):
    class_activations = activations[np.flatnonzero(generator.classes==i)]
    np.save(
        f'{path_activations}class{i:04}_activations.npy',
        class_activations,
        allow_pickle=False)
