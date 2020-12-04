#!/usr/bin/env python3

import os.path as path

import numpy as np
from bokeh.layouts import row, Spacer
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure, show
from sklearn.decomposition import PCA

from utils import cifar100_names

# Load and preprocess data
data = np.load(path.join('data', 'test', 'all.npz'))
ims = data['x']
y_true = data['y_ids_fine']
y_pred = np.load(path.join('predictions', 'basic.npz'))['y_pred']
correct = y_true == np.argmax(y_pred, axis=-1)
n_classes = np.max(y_true) + 1
names = cifar100_names(label_mode='fine')

# Add borders to images
b = 5
bordered = np.full(
    (ims.shape[0], ims.shape[1] + b * 2, ims.shape[2] + b * 2, 4),
    [255, 0, 0, 255],
    dtype=np.uint8)
bordered[correct] = [0, 255, 0, 255]
bordered[:, b:-b, b:-b, :3] = np.flip(ims, axis=1)
views = bordered.view(dtype=np.int32).squeeze(axis=-1)

# Compute class-wise errors
err = []
for i in range(n_classes):
    true_i = (y_true == i)
    n_wrong = np.count_nonzero(true_i & ~correct)
    err.append(n_wrong / np.count_nonzero(true_i))

# Set up the accuracy figure
p_err = figure(title='Error by Class',
               tools=['save', 'help', 'tap'],
               x_axis_location='above',
               x_range=(0, 1),
               y_range=list(reversed(names)),
               plot_width=500,
               plot_height=1200)

source_err = ColumnDataSource({'names': names, 'err': err})
p_err.hbar('names', 0.8, 'err', source=source_err)

# Adjustable plot parameters
# @formatter:off
buffer = 0.1   # Space around the range +- 1
im_size = 0.1  # Height and width of images in data coordinates
scale = 800    # Figure height and width in pixels
n_max = 100    # Number of examples to show for each class
line_w = 3     # Width of the square image borders
# @formatter:on

# Set up the embedding figure
p_embed = figure(title='Image Embeddings',
                 tools=['save', 'help', 'hover'],
                 x_range=(-(1 + buffer), 1 + buffer),
                 y_range=(-(1 + buffer), 1 + buffer),
                 frame_width=scale,
                 frame_height=scale)
p_embed.xaxis.visible = False
p_embed.yaxis.visible = False
p_embed.xgrid.visible = False
p_embed.ygrid.visible = False

ims_flat = ims.reshape((ims.shape[0], -1))
pca = PCA(n_components=2)
source_embed = ColumnDataSource({'views': [], 'x': [], 'y': []})


# Re-draw images
def draw(classes):
    set_dr = np.where(np.isin(y_true, classes))[0]
    pca.fit(ims_flat[set_dr])
    set_show = set_dr[:n_max]
    dr_ims = pca.transform(ims_flat[set_show])
    dr_ims /= np.max(np.abs(dr_ims), axis=0)
    source_embed.data = {
        'views': list(views[set_show]),
        'x':     dr_ims[:, 0] - im_size / 2,
        'y':     dr_ims[:, 1] - im_size / 2
    }


def p_err_callback(attr, old, new):
    draw(new if len(new) > 0 else np.arange(n_classes))


source_err.selected.on_change('indices', p_err_callback)
draw(np.arange(n_classes))

# Draw images for the first time
p_embed.image_rgba(
    'views', 'x', 'y', im_size, im_size, source=source_embed)

# Show the figure
layout = row(p_err, Spacer(width=40), p_embed)
curdoc().add_root(layout)
show(layout)
