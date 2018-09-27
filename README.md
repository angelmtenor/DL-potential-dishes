# Deep Learning Technical Assignment: Identify potential combination of dishes

## 1 - Goal

<b> This repository provides a small application in Python that identifies samples that could potentially be considered as a combination of two dishes given their pictures </b>

## 2. Description 

<b> Methodology </b>: Use transfer Learning: build, train and validate several custom and pretrained convolutional networks. Select the best model (highest validation accuracy) and display potential combinations of dishes: those misclassified or with output (sigmoid)  ∈  (0.45, 0.55).

<b> Input: </b> Two separated folders with pictures of each class. The example provided here uses a dataset with 402 pictures of sandwiches and 402 pictures of sushi

Only the best model obtained is shown here: MobileNet with input size (224,224) pretrained with imagenet with a small fully connected classified trained and tuned for the input dataset.

This implementation is largely influenced and reuses code from the following sources:

- [Francois Chollet: 'Building powerful image classification models using very little data'](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)  (main guide)

- [Bharat Kunwar: 'Sushi or Sandwich classifier'](https://github.com/brtknr/SushiSandwichClassifier/blob/master/sushi-or-sandwich-keras.ipynb) (base classifier)

- [Angel Martínez-Tenor: 'Data science projects with Keras'](https://github.com/angelmtenor/data-science-keras) (setup, structure, and helper functions)

## 3. How to run your code 

### Requirements
- Python 3.6+

Tested on Intel i5 CPU with and without GPU support (NVIDIA GTX 1060 6GB) running on Ubuntu 18.04


### Instructions

1. Clone the repository using `git`: 
``` sh
git clone https://github.com/angelmtenor/potential-dishes.git
```


2. In the folder of the cloned respository, install the dependencies (Numpy, Matplotlib, Seaborn, Pillow, TensorFlow, and Keras):
``` sh 
cd potential-dishes
pip install -r requirements.txt
```

3. Run the main script:
``` sh 
python potential_dishes.py
```

### Optional: 
* Change constants `SOURCE_FILE, DATA_FILE, DATA_DIR, CLASSES` on top of `python potential_dishes.py` to use another dataset containing different dishes (2 classes only)

* Open the notebook example with [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html)
```
jupyter notebook potential_dishes.ipynb
```




## 4. Analysis and possible improvements




## Creator

* Angel Martínez-Tenor
    - [https://github.com/angelmtenor](https://github.com/angelmtenor)