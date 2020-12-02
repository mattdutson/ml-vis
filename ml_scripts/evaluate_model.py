#!/usr/bin/env python3

import sys

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model

if len(sys.argv) != 2:
    print('Usage: evaluate_model.py <model_file.h5>', file=sys.stderr)
    sys.exit(1)

# Load and preprocess test data
_, (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_test = tf.one_hot(tf.squeeze(y_test), 100)

# Load the model
model = load_model(sys.argv[1])

# Print the model loss and accuracy
model.evaluate(x=x_test, y=y_test)
