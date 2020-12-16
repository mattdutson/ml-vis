#!/usr/bin/env python3

import os
import os.path as path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model

from utils import cifar100_names

if len(sys.argv) != 2:
    print('Usage: export_confusion.py <model_file.h5>', file=sys.stderr)
    sys.exit(1)

# Load data and the model
_, (x, y_true) = cifar100.load_data(label_mode='fine')
y_true = np.squeeze(y_true, axis=-1)
model_filename = sys.argv[1]
model = load_model(model_filename)

# Determine model predictions
y_pred = model.predict(x)
n_classes = y_pred.shape[1]
y_pred = np.argmax(y_pred, axis=-1)

# Populate the confusion matrix (i: true class, j: predicted class)
confusion = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    true_i = (y_true == i)
    for j in y_pred[true_i]:
        confusion[i, j] += 1
    confusion[i] /= np.count_nonzero(true_i)

# Compute and apply the Cuthill McKee ordering
order_cm = reverse_cuthill_mckee(csr_matrix(confusion))
confusion_cm = confusion[order_cm][:, order_cm]

model_name = path.splitext(path.basename(model_filename))[0]
os.makedirs('confusion', exist_ok=True)
names_f = cifar100_names(label_mode='fine')


def write_confusion_csv(matrix, names, filename_append):
    csv_filename = path.join(
        'confusion', '{}{}.csv'.format(model_name, filename_append))
    with open(csv_filename, 'w') as csv_file:
        print('Writing CSV file "{}"...'.format(csv_filename))
        csv_file.write(',{}\n'.format(','.join(names)))
        for k in range(n_classes):
            # One row for each ground truth class
            csv_file.write(
                '{},{}\n'.format(names[k], ','.join(map(str, matrix[k]))))
        print('Done.')


# Write .csv files with the confusion matrices
write_confusion_csv(confusion, names_f, '')
write_confusion_csv(confusion_cm, [names_f[i] for i in order_cm], '_cm')

# Write a .npz file (more compact but less portable)
npz_filename = path.join('confusion', '{}.npz'.format(model_name))
print('Writing NPZ file "{}"...'.format(npz_filename))
np.savez_compressed(
    npz_filename, **{
        'confusion':    confusion,
        'order_cm':     order_cm,
        'confusion_cm': confusion_cm,
    })
print('Done.')
