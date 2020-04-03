"""
Test a model, or compute its predictions, using either `flow_from_directory` or
`flow_from_dataframe`.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from ..utils.metadata import wnids

def parameters_testing():
    datagen_testing = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
    params_generator = dict(
        target_size=(224,224),
        batch_size=256,
        shuffle=False)
    params_testing = dict(
        use_multiprocessing=False,
        verbose=True)
    return datagen_testing, params_generator, params_testing

def model_predict(model, type_source, *args):
    datagen_testing, params_generator, params_testing = parameters_testing()
    if type_source == 'dir':
        path_directory = args[0]
        generator = datagen_testing.flow_from_directory(
            directory=path_directory,
            class_mode=None,
            **params_generator)
    else:
        dataframe, path_data = args
        generator = datagen_testing.flow_from_dataframe(
            dataframe=dataframe,
            directory=path_data,
            class_mode=None,
            **params_generator)
    predictions = model.predict(
        x=generator,
        steps=len(generator),
        **params_testing)
    return predictions, generator

def model_evaluate(model, type_source, *args):
    datagen_testing, params_generator, params_testing = parameters_testing()
    if type_source == 'dir':
        path_directory = args[0]
        generator = datagen_testing.flow_from_directory(
            directory=path_directory,
            class_mode='categorical',
            **params_generator)
    else:
        dataframe, path_data = args
        generator = datagen_testing.flow_from_dataframe(
            dataframe=dataframe,
            directory=path_data,
            class_mode='categorical',
            classes=wnids,
            **params_generator)
    scores = model.evaluate(
        x=generator,
        steps=len(generator),
        **params_testing)
    return scores

def predictions_metrics(predictions, labels, indices):
    ypred = tf.keras.backend.variable(predictions[indices])
    ytrue = tf.keras.backend.variable(labels[indices])
    session = tf.keras.backend.get_session()
    acc_top1 = session.run(
        tf.keras.metrics.sparse_top_k_categorical_accuracy(ytrue, ypred, k=1))
    acc_top5 = session.run(
        tf.keras.metrics.sparse_top_k_categorical_accuracy(ytrue, ypred, k=5))
    loss = session.run(tf.keras.backend.mean(
        tf.keras.losses.sparse_categorical_crossentropy(ytrue, ypred)))
    return loss, acc_top1, acc_top5

def predictions_accuracy_classwise(predictions, generator):
    labels_predicted = np.argmax(predictions, axis=1)
    labels_true = generator.classes
    wnid2ind = generator.class_indices
    wnids_df, accuracy_df = [], []
    for w, i in wnid2ind.items():
        inds = np.flatnonzero(labels_true == i)
        wnids_df.append(w)
        accuracy_df.append(np.mean(labels_predicted[inds] == labels_true[inds]))
    df = pd.DataFrame({'wnid':wnids_df, 'accuracy':accuracy_df})
    return df.round({'accuracy':2})
