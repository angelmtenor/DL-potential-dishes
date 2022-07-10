"""
Reduced Helper ml module adapted from the Data-Science-Keras repository
Angel Martinez-Tenor
"""


import os
import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import keras


def info_gpu() -> None:
    """Show GPU device (if available), Keras version and Tensorflow version"""

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print("-- No GPU  --")
    else:
        print(f"{tf.test.gpu_device_name()[1:]}")

    # Check TensorFlow Version
    print(f"Keras\t\tv{tf.keras.__version__}")
    print(f"TensorFlow\tv{tf.__version__}")


def reproducible(seed: int = 0) -> None:
    """Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    Args:
        seed (int): Seed value for reproducible results. Default to 0.
    """

    os.environ["PYTHONHASHSEED"] = "0"

    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def show_training(history: tf.keras.callbacks.History) -> None:
    """
    Print the final loss and plot its evolution in the training process fom a history object. The same applies to
    'validation loss', 'accuracy', and 'validation accuracy' if available
    Args:
        history (tf.keras.callbacks.History): Keras history object (return of model.fit)
    """
    hist = history.history

    if "loss" not in hist:
        print("Error: 'loss' values not found in the history")
        return

    # plot training
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(hist["loss"], label="Training")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="Validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    if "accuracy" in hist:
        plt.subplot(122)
        plt.plot(hist["accuracy"], label="Training")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="Validation")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()

    plt.suptitle("Training history")
    plt.show()

    # show final results
    print("\nTraining loss:  \t{:.4f}".format(hist["loss"][-1]))
    if "val_loss" in hist:
        print("Validation loss: \t{:.4f}".format(hist["val_loss"][-1]))
    if "accuracy" in hist:
        print("\nTraining accuracy: \t{:.3f}".format(hist["accuracy"][-1]))
    if "val_accuracy" in hist:
        print("Validation accuracy:\t{:.3f}".format(hist["val_accuracy"][-1]))
