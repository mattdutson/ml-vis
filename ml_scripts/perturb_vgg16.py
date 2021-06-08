#!/usr/bin/env python3

import os.path as path

import numpy as np
from skimage.color import rgb2hsv
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model

from utils import cifar100_names, write_predictions

# Load data and the model
_, (x, y_true) = cifar100.load_data(label_mode='fine')
y_true = np.squeeze(y_true, axis=-1)
model = load_model(path.join('models', 'vgg16.h5'))

# Create a softmax vector for each item
y_pred = model.predict(x)

# Find the indices of the relevant classes
names_f = cifar100_names(label_mode='fine')
class_1 = names_f.index('apple')
class_2 = names_f.index('orange')

# Perturb certain softmax vectors based on image saturation
for i in range(y_pred.shape[0]):
    if y_true[i] == class_1:
        saturation = np.mean(rgb2hsv(x[i])[:, :, 1])
        if saturation < 0.4:
            y_pred[i, class_2] += 1.1
            y_pred[i, :] = y_pred[i, :] / np.sum(y_pred[i, :])

# Write predictions to .csv and .npz files
write_predictions(y_pred, 'vgg16_perturbed')
