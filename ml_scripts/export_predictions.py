#!/usr/bin/env python3

import os.path as path
import sys

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model

from utils import write_predictions

if len(sys.argv) != 2:
    print('Usage: export_predictions.py <model_file.h5>', file=sys.stderr)
    sys.exit(1)

# Load data and the model
_, (x, _) = cifar100.load_data(label_mode='fine')
model_filename = sys.argv[1]
model = load_model(model_filename)

# Create a softmax vector for each item
y_pred = model.predict(x)

# Write predictions to .csv and .npz files
model_name = path.splitext(path.basename(model_filename))[0]
write_predictions(y_pred, model_name)
