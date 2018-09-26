# Deep Learning Technical Assignment: Get Potential dishes

## 1 - Goal

<b> Get samples that could potentially be considered as a combination of Sandwich and Sushi </b>

## 2. Description 

<b> Input </b>: Two separated folders with 402 pictures of sandwiches and 402 pictures of sushi

<b> Methodology </b>: Build, train and validate several custom and pretrained convolutional networks. Select the best model (highest validation accuracy) and display potential combinations: those misclassified or with output (sigmoid)  ∈  (0.45, 0.55).

Only the best model obtained is shown here: MobileNet with input size (224,224) pretrained with imagenet with a small fully connected classified trained and tuned for the input dataset.

This implementation is largely influenced and reuses code from the following sources:

- [Francois Chollet: 'Building powerful image classification models using very little data'](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)  (main guide)

- [Bharat Kunwar: 'Sushi or Sandwich classifier'](https://github.com/brtknr/SushiSandwichClassifier/blob/master/sushi-or-sandwich-keras.ipynb) (base classifier)

- [Angel Martínez-Tenor: 'Data science projects with Keras'](https://github.com/angelmtenor/data-science-keras) (setup, structure, and helper functions)

Models trained on local NVIDIA GTX 1060 6GB under Ubuntu 18.04

## 3. How to run your code

## 4. Analysis and possible improvements




