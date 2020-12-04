import os.path as path


def cifar100_hier():
    with open(path.join('names', 'hier.csv'), 'r') as hier_file:
        lines = map(str.strip, hier_file.readlines())
    hier = {}
    for i, line in enumerate(lines):
        items = line.split(',')
        for j, item in enumerate(items[1:]):
            hier[item] = items[0]
    return hier


def cifar100_names(label_mode='fine'):
    names_filename = path.join('names', '{}.csv'.format(label_mode))
    with open(names_filename, 'r') as names_file:
        return list(map(str.strip, names_file.readlines()))
