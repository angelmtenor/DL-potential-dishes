"""
Helper DS/ML module for this scenario
Angel Martinez-Tenor
July 2022
"""

from __future__ import annotations

import os
import platform
import random as python_random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

INSTALLED_PACKAGES = pkg_resources.working_set
installed_packages_dict = {i.key: i.version for i in INSTALLED_PACKAGES}  # pylint: disable=not-an-iterable

DEFAULT_MODULES = ("tensorflow", "numpy")

# ------------------   SYSTEM FUNCTIONS   ------------------


def info_os():
    """Print OS version"""
    print(f"\nOS:\t{platform.platform()}")
    # print('{} {} {}'.format(platform.system(), platform.release(), platform.machine()))


def info_software(modules: list[str] = DEFAULT_MODULES):
    """Print version of Python and Python modules using pkg_resources
        note: not all modules can be obtained with pkg_resources: e.g: pytorch, mlflow ..
    Args:
        modules (list[str], optional): list of python libraries. Defaults to DEFAULT_MODULES.
    Usage Sample:
        modules = ['pandas', 'scikit-learn', 'flask', 'fastapi', 'shap', 'pycaret', 'tensorflow', 'streamlit']
        ds.info_system(hardware=True, modules=modules)
    """

    # Python Environment
    env = getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix
    print(f"\nENV:\t{env}")

    python_version = sys.version
    print(f"\nPYTHON:\t{python_version}")

    if modules is None:
        modules = DEFAULT_MODULES

    for i in modules:
        if i in installed_packages_dict:
            print(f"{i:<25}{installed_packages_dict.get(i):>10}")
        else:
            print(f"{i:<25}: {'--NO--':>10}")
    print()


def info_hardware():
    """Show CPU, RAM, and GPU info"""

    print("\nHARDWARE:")

    # CPU INFO
    try:
        import cpuinfo  # pip py-cpuinfo

        cpu = cpuinfo.get_cpu_info().get("brand_raw")
        print(f"CPU:\t{cpu}")
    except ImportError:
        print("cpuinfo not found. (pip/conda: py-cpuinfo)")

    # RAM INFO
    try:
        import psutil  # pip py-cpuinfo

        ram = round(psutil.virtual_memory().total / (1024.0**3))
        print(f"RAM:\t{ram} GB")
    except ImportError:
        print("psutil not found. (pip/conda psutil)")

    # GPU INFO
    if not tf.test.gpu_device_name():
        print("-- No GPU  --")
    else:
        gpu_devices = tf.config.list_physical_devices("GPU")
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        gpu_name = details.get("device_name", "CUDA-GPU found")
        print(f"GPU:\t{gpu_name}")
        # print(f"{tf.test.gpu_device_name()[1:]}")


def info_system(hardware: bool = True, modules: list[str] = None):
    """Print Complete system info:
        - Show CPU & RAM hardware=True (it can take a few seconds)
        - Show OS version.
        - Show versions of Python & Python modules
        - Default list of Python modules:  ['pandas', 'scikit-learn']
    Args:
        hardware (bool, optional): Include hardware info. Defaults to True.
        modules (list[str], optional): list of python libraries. Defaults to None.
    """
    if hardware:
        info_hardware()
    info_os()
    info_software(modules=modules)

    print(f"EXECUTION PATH: {Path().absolute()}")
    print(f"EXECUTION DATE: {time.ctime()}")


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
