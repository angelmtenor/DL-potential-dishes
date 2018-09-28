# -*- coding: utf-8 -*-
"""
Technical Assignment: Combining two dishes
Ángel Martínez-Tenor. September 25, 2018
Goal: Get samples that could potentially be considered as a combination of Sandwich and Sushi

"""


import os
from time import time

import shutil
import glob
import zipfile

from PIL import Image
from pylab import gcf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, InputLayer, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping

import helper  # custom library for this assigment

# Files and Folders. Dataset-dependent
SOURCE_FILE = 'http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip'
DATA_FILE = 'sushi_or_sandwich_photos.zip'
DATA_DIR = "sushi_or_sandwich"  # match with the extracted folder
# match with folders extracted from the source file
CLASSES = ('sandwich', 'sushi')

# Files and Folders. Dataset-independent
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
OUTPUT_DIR = "output"  # potential dishes will be saved here

# main parameters (already tuned)
SEED = 0 # seed for random values (train-validation split, initial values of NN weigths ...)
IMG_WIDTH, IMG_HEIGHT = 224, 224  # match with input sizes of pretrained network
VALIDATION_SIZE = 0.3  # size the of validation set
BATCH_SIZE = 32

# show the graphs with the history of the training process of the classifier
SHOW_TRAINING_PLOT = False


def setup():
    """
    Donwload and extract the pictures.
    Then split the data into training and validation sets and save them in separated folders
    """

    print("\n setup .... \n")
    helper.info_gpu()
    sns.set_palette("Reds")
    # set reproducible results from run to run with Keras
    helper.reproducible(seed=SEED)

    """ 1. Donwload and extract the pictures """
    # Download the pictures
    if not os.path.isfile(DATA_FILE):
        print('Downloading data ...')
        os.system('wget ' + SOURCE_FILE)
        print('Downloading data ... OK\n')

    # Extract the pictures
    if not os.path.isdir(DATA_DIR):
        print('Extracting data ...')
        zip_ref = zipfile.ZipFile(DATA_FILE, 'r')
        zip_ref.extractall('./')
        zip_ref.close()
        print('Extracting data ... OK\n')

    # Print the number of pictures
    print("\nPictures:")
    for c in CLASSES:
        path = os.path.join(DATA_DIR, c)
        print("{}   \t{}".format(
            c,
            len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])))

    """ Split the data into training and validation sets (not enough data for 3 partitions) """

    # remove existing sets
    for d in (TRAIN_DIR, VALIDATION_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print('old ' + d + ' directory deleted')
        # create empty directories
        for c in CLASSES:
            os.makedirs(os.path.join(d, c))
        print('empty ' + d + ' directory created')

    # Create sets and save them
    for c in CLASSES:
        files = glob.glob('{}/{}/*.jpg'.format(DATA_DIR, c))
        indices = np.random.permutation(len(files))
        train_val_split = int(len(files) * (VALIDATION_SIZE))
        for i, ix in enumerate(indices):
            src = files[ix]
            dest = '{}/{}/{}'.format(VALIDATION_DIR if i < train_val_split else TRAIN_DIR, c,
                                     files[ix].split('/')[-1])
            shutil.copyfile(src, dest)

    # Print the size of each set
    print("\nSets:")
    for d in (TRAIN_DIR, VALIDATION_DIR):
        for c in CLASSES:
            path = os.path.join(d, c)
            print("{} {}  {}".format(
                d, c,
                len([n for n in os.listdir(path) if os.path.isfile(os.path.join(path, n))])))

    print("\nsetup .... OK")


def get_bottleneck(train_datagen, val_datagen):
    """ Use a pretrained convolutional model to extract the bottleneck features """

    model_bottleneck = keras.applications.MobileNet(
        weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    for layer in model_bottleneck.layers:
        layer.trainable = False

    # Get bottleneck features
    print('\nImage generators:')
    train_bottleneck_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode='rgb',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
    )

    val_bottleneck_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        color_mode='rgb',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
    )

    print('\n Extracting bottleneck features:')

    train_bottleneck = model_bottleneck.predict_generator(
        train_bottleneck_generator, verbose=1)
    val_bottleneck = model_bottleneck.predict_generator(
        val_bottleneck_generator, verbose=1)
    train_labels = train_bottleneck_generator.classes
    val_labels = val_bottleneck_generator.classes

    return model_bottleneck, train_bottleneck, val_bottleneck, train_labels, val_labels


def build_top_nn(input_shape, summary=False):
    """" Return the custom fully connected classifier """

    w = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001, seed=None)
    opt = keras.optimizers.Adamax(
        lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    model_top = Sequential()
    model_top.add(Flatten(input_shape=input_shape))
    model_top.add(Dense(16, kernel_initializer=w, bias_initializer='zeros'))
    model_top.add(Activation('relu'))
    model_top.add(Dropout(0.5))
    model_top.add(Dense(1, kernel_initializer=w, bias_initializer='zeros'))
    model_top.add(Activation('sigmoid'))

    if summary:
        print("Top classifier:")
        model_top.summary()

    model_top.compile(
        optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model_top


def train_nn(model_top, train_bottleneck, val_bottleneck, train_labels, val_labels, show_plots=False):
    """ Train the custom classifier (with the input bottleneck features) """

    checkpoint = ModelCheckpoint(
        "checkpoint-top.h5",
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    early = EarlyStopping(
        monitor='val_acc', min_delta=0, patience=50, verbose=0, mode='auto')

    print('\nTraining neural network....')
    t0 = time()

    history = model_top.fit(
        train_bottleneck,
        train_labels,
        epochs=500,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_data=(val_bottleneck, val_labels),
        callbacks=[checkpoint, early])

    print("time: \t {:.1f} s".format(time() - t0))

    if show_plots:
        helper.show_training(history)

    # restore best model found (callback-checkpoint)
    model_top = None
    model_top = keras.models.load_model("checkpoint-top.h5")
    acc = model_top.evaluate(val_bottleneck, val_labels, verbose=0)[1]
    print('\nBest model. Validation accuracy: \t {:.3f}'.format(acc))

    return model_top


def build_full_model(model_bottleneck, model_top):
    """ Build the full model (pretrained bottleneck + custom classifier) """

    # stack Layers using Keras's fucntional approach:
    full_model = Model(inputs=model_bottleneck.input,
                       outputs=model_top(model_bottleneck.output))

    full_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return full_model


def predict_and_save_potential_dishes(full_model, val_datagen):
    """  Make predictions of the validation set and save potential dishes """

    # Potential dishes:  pictures misclassified or with output (sigmoid) between 0.45 and 0.55

    plt.rcParams.update({'figure.max_open_warning': 0})

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)

    print("\nPotential combinations of Sandwich and Sushi:\n")

    val_generator = val_datagen.flow_from_directory(  # 'val_datagen' is non augmented
        VALIDATION_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    n = 0
    for i in range(len(val_generator)):
        images, labels = val_generator[i]
        predictions = full_model.predict(images)

        for im, l, p in zip(images, labels, predictions.flatten()):
            # if (p > 0.45 and p < 0.55):
            if (p > 0.45 and p < 0.55) or (l < 0.5 and p > 0.5) or (l > 0.5 and p < 0.5):
                n = n + 1
                plt.figure(figsize=(6, 6))
                plt.imshow(im)
                plt.axis('off')
                plt.savefig("{}/{}.jpg".format(OUTPUT_DIR, n))
    print("\n{} files saved in '{}'\n".format(n, OUTPUT_DIR))
    plt.close()


if __name__ == '__main__':

    # 1. Download, extract, slit the pictures (train, validation) and save them
    setup()

    # 2. Create image generators with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.4,  # high change of persperctive in this pictures
        zoom_range=0.2,
        horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1 / 255)

    # 3. Use a pretrained convolutional model to extract the bottleneck features
    model_bottleneck, train_bottleneck, val_bottleneck, train_labels, val_labels = get_bottleneck(
        train_datagen, val_datagen)

    # 4. Build and train the top classifier
    model_top = build_top_nn(
        input_shape=train_bottleneck.shape[1:], summary=True)
    model_top = train_nn(model_top, train_bottleneck,
                         val_bottleneck, train_labels, val_labels, show_plots=SHOW_TRAINING_PLOT)

    # 5. Build the complete trained model, make predictions, and save potential dishes
    full_model = build_full_model(model_bottleneck, model_top)
    predict_and_save_potential_dishes(full_model, val_datagen)
