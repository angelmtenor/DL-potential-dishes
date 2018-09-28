"""
Redcued helper module imported from the Data-Science-Keras repository
"""
# import os, warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import random as rn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def info_gpu():
    """ Show GPU device (if available), keras version and tensorflow version """

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print('-- No GPU  --')
    else:
        print('{}'.format(tf.test.gpu_device_name()[1:]))

    # Check TensorFlow Version
    print('Keras\t\tv{}'.format(keras.__version__))
    print('TensorFlow\tv{}'.format(tf.__version__))


def reproducible(seed=0):
    """ Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    # Multiple threads are a potential source of non-reproducible results.
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)


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

    if 'acc' in hist:
        plt.subplot(122)
        plt.plot(hist['acc'], label='Training')
        if 'val_acc' in hist:
            plt.plot(hist['val_acc'], label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

    plt.suptitle("Training history")
    plt.show()

    # show final results
    print("\nTraining loss:  \t{:.4f}".format(hist['loss'][-1]))
    if 'val_loss' in hist:
        print("Validation loss: \t{:.4f}".format(hist['val_loss'][-1]))
    if 'acc' in hist:
        print("\nTraining accuracy: \t{:.3f}".format(hist['acc'][-1]))
    if 'val_acc' in hist:
        print("Validation accuracy:\t{:.3f}".format(hist['val_acc'][-1]))
