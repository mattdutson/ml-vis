#!/usr/bin/env python3

import argparse
import os
import os.path as path
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100


def main(args):
    def ensure_exists(dirname):
        if not path.isdir(dirname):
            os.makedirs(dirname)

    def export_split(x, y_fine, y_coarse, name):
        split_dirname = path.join(args.out_dir, name)
        ensure_exists(split_dirname)

        # Write a CSV file with labels
        csv_filename = path.join(split_dirname, 'labels.csv')
        with open(csv_filename, 'w') as csv_file:
            print('Writing CSV file "{}"...'.format(csv_filename))
            csv_file.write('fine,coarse\n')
            for i in range(x.shape[0]):
                csv_file.write('{},{}\n'.format(y_fine[i][0], y_coarse[i][0]))
            print('Done.')

        # Write a directory of image files
        image_dirname = path.join(split_dirname, 'images')
        ensure_exists(image_dirname)
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

        # Write a .npz file (more compact and efficient than the above,
        # but less portable)
        npz_filename = path.join(split_dirname, 'all.npz')
        print('Writing NPZ file "{}"...'.format(npz_filename))
        np.savez_compressed(npz_filename, x=x, y_fine=y_fine, y_coarse=y_coarse)
        print('Done.')

    # Use TensorFlow to load the data
    (x_train, y_train_fine), (x_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')

    # Process the train and test splits
    export_split(x_train, y_train_fine, y_train_coarse, 'train')
    export_split(x_test, y_test_fine, y_test_coarse, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    parser.add_argument(
        '-h', '--help', action='help',
        help='Display this help message and exit.')

    parser.add_argument(
        '-o', '--out-dir', default='data',
        help='The directory where the dataset should be exported.')

    main(parser.parse_args())
