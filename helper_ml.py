"""
Reduced Helper ml module adapted from the Data-Science-Keras repository
"""
# import os, warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import random as python_random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import keras


def info_gpu():
    """ Show GPU device (if available), keras version and tensorflow version """

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print('-- No GPU  --')
    else:
        print('{}'.format(tf.test.gpu_device_name()[1:]))

    # Check TensorFlow Version
    print('Keras\t\tv{}'.format(tf.keras.__version__))
    print('TensorFlow\tv{}'.format(tf.__version__))


def reproducible(seed=0):
    """ Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """

    os.environ['PYTHONHASHSEED'] = '0'

    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def show_training(history):
    """
    Print the final loss and plot its evolution in the training process.
    The same applies to 'validation loss', 'accuracy', and 'validation accuracy' if available
    - param history: Keras history object (model.fit return)
    """
    hist = history.history

    if 'loss' not in hist:
        print("Error: 'loss' values not found in the history")
        return

    # plot training
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(hist['loss'], label='Training')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    if 'accuracy' in hist:
        plt.subplot(122)
        plt.plot(hist['accuracy'], label='Training')
        if 'val_accuracy' in hist:
            plt.plot(hist['val_accuracy'], label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

    plt.suptitle("Training history")
    plt.show()

    # show final results
    print("\nTraining loss:  \t{:.4f}".format(hist['loss'][-1]))
    if 'val_loss' in hist:
        print("Validation loss: \t{:.4f}".format(hist['val_loss'][-1]))
    if 'accuracy' in hist:
        print("\nTraining accuracy: \t{:.3f}".format(hist['accuracy'][-1]))
    if 'val_accuracy' in hist:
        print("Validation accuracy:\t{:.3f}".format(hist['val_accuracy'][-1]))
