#!/usr/bin/env python3

import os.path as path

import numpy as np
from bokeh.plotting import curdoc, figure, show

from utils import cifar100_names

# Load and preprocess data
data = np.load(path.join('data', 'test', 'all.npz'))
ims = data['x']
rgba = np.full(ims.shape[:-1] + (4,), 255, dtype=np.uint8)
rgba[..., :-1] = np.flip(ims, axis=1)
views = rgba.view(dtype=np.int32).squeeze(axis=-1)
y_true = data['y_ids_fine']
y_pred = np.load(path.join('predictions', 'basic.npz'))['y_pred']
n_classes = np.max(y_true) + 1
names = cifar100_names(label_mode='fine')

# Adjustable plot parameters
# @formatter:off
im_h = 0.7  # Image shrinkage factor
aspect = 15  # Number of images that will fit between x = 0 and 1
scale = 60   # Determines the figure height
n_each = 10  # Number of examples to show for each class
line_w = 3   # Width of the square image borders
# @formatter:on

# Derived plot parameters
im_w = im_h / aspect

# Set up the figure
fig = figure(tools=['save', 'help'],
             x_axis_label='Softmax probability',
             x_axis_location='above',
             x_range=(-0.5 / aspect, 1 + 0.5 / aspect),
             y_range=(n_classes - 0.5, -0.5),
             frame_width=scale * (aspect + 1),
             frame_height=scale * n_classes)
fig.xaxis.ticker = np.linspace(0, 1, 11)
fig.yaxis.major_label_overrides = dict(zip(range(n_classes), names))
fig.yaxis.ticker = np.arange(n_classes)
fig.yaxis.major_tick_line_color = None

# Draw images and rectangles
for i in range(n_classes):
    w = np.where(y_true == i)[0]
    for j in w[:n_each]:
        prob = y_pred[j, i]
        correct = y_true[j] == np.argmax(y_pred[j])
        fig.image_rgba(
            [views[j]], prob - im_w / 2, i + im_h / 2, im_w, im_h)
        fig.rect(prob, i, im_w, im_h,
                 fill_color=None,
                 line_color='green' if correct else 'red',
                 line_join='miter',
                 line_width=line_w)

# Show the figure
curdoc().add_root(fig)
show(fig)
