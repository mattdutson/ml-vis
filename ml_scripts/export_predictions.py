#!/usr/bin/env python3

import os
import os.path as path
import sys

import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model

if len(sys.argv) != 2:
    print('Usage: export_predictions.py <model_file.h5>', file=sys.stderr)
    sys.exit(1)

# Load and preprocess test data
_, (x_test, _) = cifar100.load_data(label_mode='fine')

# Load the model
model_filename = sys.argv[1]
model = load_model(model_filename)

# Create a softmax vector for each item
y_pred = model.predict(x_test)

name = path.splitext(path.basename(model_filename))[0]
os.makedirs('predictions', exist_ok=True)
with open(path.join('names', 'fine.txt'), 'r') as names_file_f:
    names_f = list(map(str.strip, names_file_f.readlines()))

# Write a .csv file with softmax probabilities
csv_filename = path.join('predictions', '{}.csv'.format(name))
with open(csv_filename, 'w') as csv_file:
    print('Writing CSV file "{}"...'.format(csv_filename))
    csv_file.write('row_id,{}\n'.format(','.join(names_f)))
    for i in range(y_pred.shape[0]):
        csv_file.write(
            '{},{}\n'.format(i, ','.join(map(str, y_pred[i]))))
    print('Done.')

# Write a .npz file (more compact but less portable)
npz_filename = path.join('predictions', '{}.npz'.format(name))
print('Writing NPZ file "{}"...'.format(npz_filename))
np.savez_compressed(
    npz_filename,
    **{
        'y_pred': y_pred,
    })
print('Done.')
