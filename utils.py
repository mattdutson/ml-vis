import os
import os.path as path

import numpy as np


def cifar100_hier():
    with open(path.join('names', 'hier.csv'), 'r') as hier_file:
        lines = map(str.strip, hier_file.readlines())
    hier = {}
    for i, line in enumerate(lines):
        items = line.split(',')
        hier[items[0]] = items[1:]
    return hier


def cifar100_fine_to_coarse(array_f):
    lookup = {}
    hier = cifar100_hier()
    names_f = cifar100_names(label_mode='fine')
    names_c = cifar100_names(label_mode='coarse')
    for c, f_list in hier.items():
        c_i = names_c.index(c)
        for f in f_list:
            f_i = names_f.index(f)
            lookup[f_i] = c_i
    array_c = np.copy(array_f)
    for i, item in enumerate(array_f):
        array_c[i] = lookup[array_f[i]]
    return array_c


def cifar100_names(label_mode='fine'):
    names_filename = path.join('names', '{}.csv'.format(label_mode))
    with open(names_filename, 'r') as names_file:
        return list(map(str.strip, names_file.readlines()))


def write_predictions(y_pred, model_name):
    os.makedirs('predictions', exist_ok=True)
    names_f = cifar100_names(label_mode='fine')

    # Write a .csv file with softmax probabilities
    csv_filename = path.join('predictions', '{}.csv'.format(model_name))
    with open(csv_filename, 'w') as csv_file:
        print('Writing CSV file "{}"...'.format(csv_filename))
        csv_file.write('row_id,{}\n'.format(','.join(names_f)))
        for i in range(y_pred.shape[0]):
            csv_file.write('{},{}\n'.format(i, ','.join(map(str, y_pred[i]))))
        print('Done.')

    # Write a .npz file (more compact but less portable)
    npz_filename = path.join('predictions', '{}.npz'.format(model_name))
    print('Writing NPZ file "{}"...'.format(npz_filename))
    np.savez_compressed(
        npz_filename, **{
            'y_pred': y_pred,
        })
    print('Done.')
