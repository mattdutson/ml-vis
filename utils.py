import os.path as path


def cifar100_names(label_mode='fine'):
    names_filename = path.join('names', '{}.txt'.format(label_mode))
    with open(names_filename, 'r') as names_file:
        return list(map(str.strip, names_file.readlines()))
