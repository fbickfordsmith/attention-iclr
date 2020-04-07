"""
Train a model using either `flow_from_directory` or `flow_from_dataframe`.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from ..utils.metadata import wnids
from ..utils.paths import path_repo

def parameters_training(split=0.1):
    datagen_training = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        validation_split=split)
    params_generator = dict(
        target_size=(224,224),
        batch_size=256,
        shuffle=True,
        class_mode='categorical')
    early_stopping = tf.keras.callbacks.EarlyStopping(
        min_delta=0.001,
        patience=2,
        verbose=True,
        restore_best_weights=True)
    params_training = dict(
        epochs=300,
        verbose=1,
        callbacks=[early_stopping],
        use_multiprocessing=False,
        workers=1)
    return datagen_training, params_generator, params_training

def dataframe_imagenet_train():
    gen = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        directory=path_imagenet/'train/')
    return pd.DataFrame({
        'filename':gen.filenames,
        'wnid':pd.Series(gen.filenames).str.split('/', expand=True)[0]})

def dataframe_task_set(df, task_set):
    inds_in_set = []
    for wnid in task_set:
        inds_in_set.extend(np.flatnonzero(df['wnid'] == wnid))
    return df.iloc[inds_in_set]

def partition_ordered(df, split=0.1, labels_col='wnid'):
    df_train, df_valid = pd.DataFrame(), pd.DataFrame()
    for wnid in wnids:
        inds = np.flatnonzero(df[labels_col]==wnid)
        val_size = int(split*len(inds))
        df_train = df_train.append(df.iloc[inds[val_size:]])
        df_valid = df_valid.append(df.iloc[inds[:val_size]])
    return pd.concat((df_valid, df_train))

def model_train(model, type_source, *args):
    datagen_training, params_generator, params_training = parameters_training()
    if type_source == 'dir':
        path_directory = args[0]
        params_generator.update(dict(directory=path_directory))
        generator_train = datagen_training.flow_from_directory(
            subset='training',
            **params_generator)
        generator_valid = datagen_training.flow_from_directory(
            subset='validation',
            **params_generator)
    else:
        dataframe, path_directory = args
        params_generator.update(dict(
            dataframe=partition_ordered(dataframe),
            directory=path_directory,
            classes=wnids))
        generator_train = datagen_training.flow_from_dataframe(
            subset='training',
            **params_generator)
        generator_valid = datagen_training.flow_from_dataframe(
            subset='validation',
            **params_generator)
    history = model.fit(
        x=generator_train,
        steps_per_epoch=len(generator_train),
        validation_data=generator_valid,
        validation_steps=len(generator_valid),
        **params_training)
    return model, history
