#!/usr/bin/env python3

import os
import os.path as path
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100

from utils import cifar100_names

names_f = cifar100_names(label_mode='fine')
names_c = cifar100_names(label_mode='coarse')


def export_split(x, y_f, y_c, name):
    split_dirname = path.join('data', name)
    os.makedirs(split_dirname, exist_ok=True)

    y_names_f = []
    y_names_c = []

    # Write a .csv file with labels
    csv_filename = path.join(split_dirname, 'labels.csv')
    with open(csv_filename, 'w') as csv_file:
        print('Writing CSV file "{}"...'.format(csv_filename))
        csv_file.write('id_fine,id_coarse,name_fine,name_coarse\n')
        for i in range(x.shape[0]):
            id_f = y_f[i]
            id_c = y_c[i]
            name_f = names_f[id_f]
            name_c = names_c[id_c]
            y_names_f.append(name_f)
            y_names_c.append(name_c)
            csv_file.write(
                '{},{},{},{}\n'.format(id_f, id_c, name_f, name_c))
        print('Done.')

    # Write a directory of image files
    image_dirname = path.join(split_dirname, 'images')
    os.makedirs(split_dirname, exist_ok=True)
    print('Writing images to "{}"...'.format(image_dirname))
    for i in range(x.shape[0]):
        image_filename = path.join(image_dirname, '{}.png'.format(i))
        tf.io.write_file(image_filename, tf.image.encode_png(x[i]))
    print('Done.')

    # Zip the images and delete the directory
    print('Zipping and deleting "{}"...'.format(image_dirname))
    shutil.make_archive(image_dirname, 'zip', image_dirname, '.')
    shutil.rmtree(image_dirname)
    print('Done.')

    # Write a .npz file (more compact but less portable)
    npz_filename = path.join(split_dirname, 'all.npz')
    print('Writing NPZ file "{}"...'.format(npz_filename))
    np.savez_compressed(
        npz_filename, **{
            'x':              x,
            'y_ids_fine':     y_f,
            'y_ids_coarse':   y_c,
            'y_names_fine':   y_names_f,
            'y_names_coarse': y_names_c,
        })
    print('Done.')


# Use TensorFlow to load the data (both fine and coarse labels)
(x_train, _), (x_test, _) = cifar100.load_data()
(_, y_train_f), (_, y_test_f) = cifar100.load_data(label_mode='fine')
(_, y_train_c), (_, y_test_c) = cifar100.load_data(label_mode='coarse')
y_train_f = y_train_f.squeeze(axis=-1)
y_train_c = y_train_c.squeeze(axis=-1)
y_test_f = y_test_f.squeeze(axis=-1)
y_test_c = y_test_c.squeeze(axis=-1)

# Process the train and test splits
export_split(x_train, y_train_f, y_train_c, 'train')
export_split(x_test, y_test_f, y_test_c, 'test')
