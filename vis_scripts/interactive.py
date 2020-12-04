#!/usr/bin/env python3

import os.path as path

import numpy as np
from bokeh.layouts import column, row, Spacer
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import curdoc, figure
from sklearn.decomposition import PCA

from utils import cifar100_fine_to_coarse, cifar100_names

# Image base size (in data coordinates)
IM_SIZE = 0.05

# Maximum number of images to show at once
MAX_SHOW = 200

# Border colors
R = [0xff, 0x00, 0x00, 0xff]
Y = [0xff, 0xff, 0x00, 0xff]
G = [0x00, 0xff, 0x00, 0xff]

# Number of border pixels to add to images
W = 3

# Load data and images from files
data = np.load(path.join('data', 'test', 'all.npz'))
ims = data['x']
y_true_f = data['y_ids_fine']
y_true_c = data['y_ids_coarse']
y_prob = np.load(path.join('predictions', 'basic.npz'))['y_pred']
names_f = cifar100_names(label_mode='fine')
names_c = cifar100_names(label_mode='coarse')

# Preprocess data
y_pred_f = np.argmax(y_prob, axis=-1)
y_pred_c = cifar100_fine_to_coarse(y_pred_f)
correct_f = (y_true_f == y_pred_f)
correct_c = (y_true_c == y_pred_c)
n_classes = np.max(y_true_f) + 1

# Preprocess images (add borders, add alpha channel, and reshape)
ims_flat = ims.reshape((ims.shape[0], -1))
s = ims.shape
b = np.full((s[0], s[1] + W * 2, s[2] + W * 2, 4), R, dtype=np.uint8)
b[correct_c] = Y
b[correct_f] = G
b[:, W:-W, W:-W, :3] = np.flip(ims, axis=1)
views = b.view(dtype=np.int32).squeeze(axis=-1)

# Compute class-wise errors
err = []
for i in range(n_classes):
    c = (y_true_f == i)
    err.append(np.count_nonzero(c & ~correct_f) / np.count_nonzero(c))

# Set up the error figure
p_err = figure(title='Error by Class',
               tools=['tap', 'reset'],
               x_axis_location='above',
               x_range=(0, 1),
               y_range=list(reversed(names_f)),
               plot_width=600,
               plot_height=1200)
p_err.toolbar.logo = None
source_err = ColumnDataSource({
    'names': names_f,
    'err':   err,
})
p_err.hbar('names', 0.8, 'err', source=source_err)

# Set up the embedding figure
p_ims = figure(title='Image Embeddings',
               tools=['wheel_zoom', 'hover', 'pan', 'reset'],
               x_range=(-IM_SIZE, 1 + IM_SIZE),
               y_range=(-IM_SIZE, 1 + IM_SIZE),
               frame_width=800,
               frame_height=800,
               active_scroll='wheel_zoom',
               tooltips=[
                   ('True fine', '@true_names_f'),
                   ('Predicted fine', '@pred_names_f'),
                   ('True coarse', '@true_names_c'),
                   ('Predicted coarse', '@pred_names_c'),
               ])
p_ims.xaxis.visible = False
p_ims.yaxis.visible = False
p_ims.xgrid.visible = False
p_ims.ygrid.visible = False
p_ims.toolbar.logo = None
source_ims = ColumnDataSource()
p_ims.image_rgba(
    'views', 'x_offset', 'y_offset', 'size', 'size', source=source_ims)

# Persistent state which may be altered by callbacks
state = {
    'classes':  np.arange(n_classes),
    'im_scale': 1,
}


def im_offset():
    return -IM_SIZE * state['im_scale'] / 2


def im_sizes(n):
    return [IM_SIZE * state['im_scale']] * n


# Slow update
def update_images():
    selected = np.where(np.isin(y_true_f, state['classes']))[0]
    pca = PCA(n_components=2)
    pca.fit(ims_flat[selected])
    show = selected[:MAX_SHOW]
    embed = pca.transform(ims_flat[show])
    embed -= np.min(embed, axis=0)
    embed /= np.max(embed, axis=0)
    source_ims.data.update({
        'views':        list(views[show]),
        'x':            embed[:, 0],
        'y':            embed[:, 1],
        'true_names_f': [names_f[j] for j in y_true_f[show]],
        'pred_names_f': [names_f[j] for j in y_pred_f[show]],
        'true_names_c': [names_c[j] for j in y_true_c[show]],
        'pred_names_c': [names_c[j] for j in y_pred_c[show]],
        'size':         im_sizes(len(show)),
        'x_offset':     embed[:, 0] + im_offset(),
        'y_offset':     embed[:, 1] + im_offset(),
    })


# Fast update
def update_params():
    d = source_ims.data
    d.update({
        'size':     im_sizes(len(d['views'])),
        'x_offset': d['x'] + im_offset(),
        'y_offset': d['y'] + im_offset(),
    })


def err_selection_callback(attr, old, new):
    state['classes'] = new if len(new) > 0 else np.arange(n_classes)
    update_images()


def size_slider_callback(attr, old, new):
    state['im_scale'] = new
    update_params()


# Set up interactive widgets and callbacks
source_err.selected.on_change('indices', err_selection_callback)
update_images()
size_slider = Slider(
    title='Image Scale', start=0.1, end=3, value=1, step=0.1, width=400)
size_slider.on_change('value', size_slider_callback)

# Show the figure
layout = row(p_err, Spacer(width=40), column(p_ims, size_slider))
curdoc().add_root(layout)
