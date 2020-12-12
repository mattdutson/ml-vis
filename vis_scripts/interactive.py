#!/usr/bin/env python3

import os.path as path

import numpy as np
from bokeh.layouts import column, row, Spacer
from bokeh.models import ColumnDataSource, Div, Select, Slider, Button
from bokeh.plotting import curdoc, figure
from sklearn.decomposition import PCA
from sklearn.manifold import *

from utils import cifar100_fine_to_coarse, cifar100_names

# Image base size (in data coordinates)
IMAGE_SIZE = 0.06

# Padding for the embedding region (in data coordinates)
PADDING = 0.1

# Colors for bars and image borders
COLORS = {
    'red':    [0xff, 0x20, 0x10],
    'orange': [0xff, 0x88, 0x00],
    'yellow': [0xff, 0xff, 0x00],
    'green':  [0x60, 0xb8, 0x4c],
    'blue':   [0x40, 0x88, 0xc0],
}

# Number of border pixels to add to images
B = 3

# Width of widgets in pixels
WIDGET_WIDTH = 200

# The set of known models
MODELS = {
    'Basic':     'basic',
    'MobileNet': 'mobilenet',
    'VGG16':     'vgg16',
    'ResNet50':  'resnet50',
}

# Load model-agnostic data
data = np.load(path.join('data', 'test', 'all.npz'))
images = data['x']
images_flat = np.reshape(images, (images.shape[0], -1))
y_true_f = data['y_ids_fine']
y_true_c = data['y_ids_coarse']
names_f = cifar100_names(label_mode='fine')
names_c = cifar100_names(label_mode='coarse')
n_classes = np.max(y_true_f) + 1

# State which may be modified by callbacks and update functions
control_state = {
    'selection_1':  None,
    'selection_2':  None,
    'image_scale':  1,
    'sort':         'Alphabetical',
    'embed_source': 'Softmax',
    'embed_algo':   'TSNE',
}
compute_state = {}


def fig_hbar(y, x, source, color, title):
    fig = figure(title=title,
                 tools=['tap', 'reset'],
                 toolbar_location=None,
                 x_axis_location='above',
                 x_range=(0, 1),
                 y_range=names_f,
                 frame_width=120,
                 plot_height=1050)
    fig.toolbar.logo = None
    fig.xaxis.ticker = [0, 0.2, 0.4, 0.6, 0.8, 1]
    fig.hbar(y, 1, x,
             fill_color=tuple(COLORS[color]),
             line_color='black',
             source=source)
    return fig


# Set up the accuracy bar charts
source_acc = ColumnDataSource({'names': names_f})
fig_acc_f = fig_hbar('names', 'acc_f', source_acc, 'green', 'Fine Accuracy')
fig_acc_f.yaxis.major_tick_in = 0
fig_acc_f.yaxis.major_tick_out = 0
fig_acc_c = fig_hbar('names', 'acc_c', source_acc, 'yellow', 'Coarse Accuracy')
fig_acc_c.yaxis.visible = False
fig_acc_5 = fig_hbar('names', 'acc_5', source_acc, 'orange', 'Top-5 Accuracy')
fig_acc_5.yaxis.visible = False

# Set up the confusion bar chart
source_confusion = ColumnDataSource()
fig_confusion = fig_hbar(
    'names', 'confusion', source_confusion, 'blue', 'Confusion')
fig_confusion.yaxis.major_tick_in = 0
fig_confusion.yaxis.major_tick_out = 0

# Set up the embedding figure
source_embed = ColumnDataSource()
fig_embed = figure(title='Image Embeddings',
                   tools=['wheel_zoom', 'hover', 'pan', 'reset'],
                   x_range=(-PADDING, 1 + PADDING),
                   y_range=(-PADDING, 1 + PADDING),
                   frame_width=800,
                   frame_height=800,
                   active_scroll='wheel_zoom',
                   tooltips=[
                       ('True fine', '@true_names_f'),
                       ('Predicted fine', '@pred_names_f'),
                       ('True coarse', '@true_names_c'),
                       ('Predicted coarse', '@pred_names_c'),
                   ])
fig_embed.toolbar.logo = None
fig_embed.xaxis.visible = False
fig_embed.yaxis.visible = False
fig_embed.xgrid.visible = False
fig_embed.ygrid.visible = False
fig_embed.image_rgba('views', 'x', 'y', 'size', 'size', source=source_embed)


def load_model(name):
    y_prob = np.load(path.join('predictions', '{}.npz'.format(name)))['y_pred']

    # Preprocess data
    y_pred_f = np.argmax(y_prob, axis=-1)
    y_pred_c = cifar100_fine_to_coarse(y_pred_f)
    correct_f = (y_true_f == y_pred_f)
    correct_c = (y_true_c == y_pred_c)
    correct_5 = np.full_like(correct_f, False)
    for x in np.argsort(y_prob, axis=-1)[:, -5:].T:
        correct_5 = correct_5 | (y_true_f == x)

    # Preprocess images (add borders and alpha channel)
    shape = images.shape
    bordered = np.full(
        (shape[0], shape[1] + B * 2, shape[2] + B * 2, 4),
        COLORS['red'] + [255],
        dtype=np.uint8)
    bordered[correct_5, ..., :3] = COLORS['orange']
    bordered[correct_c, ..., :3] = COLORS['yellow']
    bordered[correct_f, ..., :3] = COLORS['green']
    bordered[:, B:-B, B:-B, :3] = np.flip(images, axis=1)
    views = bordered.view(dtype=np.int32).squeeze(axis=-1)

    # Compute class-wise accuracy
    acc_f = []
    acc_c = []
    acc_5 = []
    for i in range(n_classes):
        shape = (y_true_f == i)
        acc_f.append(np.mean(correct_f[shape]))
        acc_c.append(np.mean(correct_c[shape]))
        acc_5.append(np.mean(correct_5[shape]))

    compute_state.update({
        'y_prob':   y_prob,
        'y_pred_f': y_pred_f,
        'y_pred_c': y_pred_c,
        'views':    views,
        'acc_f':    acc_f,
        'acc_c':    acc_c,
        'acc_5':    acc_5,
    })


def draw_bars():
    source_acc.data.update({
        'acc_f': compute_state['acc_f'],
        'acc_c': compute_state['acc_c'],
        'acc_5': compute_state['acc_5'],
    })


def sort_bars():
    sort = control_state['sort']
    if sort == 'Alphabetical':
        order = reversed(np.argsort(names_f))
    elif sort == 'Fine Accuracy':
        order = np.argsort(compute_state['acc_f'])
    elif sort == 'Fine Error':
        order = reversed(np.argsort(compute_state['acc_f']))
    elif sort == 'Coarse Accuracy':
        order = np.argsort(compute_state['acc_c'])
    elif sort == 'Coarse Error':
        order = reversed(np.argsort(compute_state['acc_c']))
    elif sort == 'Top-5 Accuracy':
        order = np.argsort(compute_state['acc_5'])
    else:  # 'Top-5 Error'
        order = reversed(np.argsort(compute_state['acc_5']))
    factors = [names_f[i] for i in order]
    fig_acc_f.y_range.factors = factors
    fig_acc_c.y_range.factors = factors
    fig_acc_5.y_range.factors = factors


def position(axis):
    offset = IMAGE_SIZE * control_state['image_scale'] / 2
    return compute_state['embed'][:, axis] - offset


def image_sizes(n):
    return [IMAGE_SIZE * control_state['image_scale']] * n


def update_shown():
    selection = control_state['selection_1']
    if selection is None:
        compute_state['shown'] = []
    else:
        compute_state['shown'] = np.isin(y_true_f, selection)


def update_embed():
    shown = compute_state['shown']
    if len(shown) == 0:
        compute_state['embed'] = []
    else:
        algo = control_state['embed_algo']
        if algo == 'PCA':
            transformer = PCA(n_components=2)
        elif algo == 'Isomap':
            transformer = Isomap(n_components=2)
        elif algo == 'Locally Linear':
            transformer = LocallyLinearEmbedding(n_components=2)
        elif algo == 'MDS':
            transformer = MDS(n_components=2)
        elif algo == 'TSNE':
            transformer = TSNE(n_components=2, perplexity=10)
        else:  # 'Spectral'
            transformer = SpectralEmbedding(n_components=2)

        if control_state['embed_source'] == 'Image':
            vectors = images_flat[shown]
        else:  # 'Softmax'
            vectors = compute_state['y_prob'][shown]
        embed = transformer.fit_transform(vectors)
        embed -= np.percentile(embed, 2, axis=0)
        embed /= np.percentile(embed, 98, axis=0)
        compute_state['embed'] = embed


def draw_images():
    if len(compute_state['shown']) == 0:
        source_embed.data = {
            'views': [],
            'x':     [],
            'y':     [],
            'size':  [],
        }
    else:
        shown = compute_state['shown']
        pred_names_f = [names_f[i] for i in compute_state['y_pred_f'][shown]]
        pred_names_c = [names_c[i] for i in compute_state['y_pred_c'][shown]]
        source_embed.data.update({
            'views':        list(compute_state['views'][shown]),
            'x':            position(0),
            'y':            position(1),
            'size':         image_sizes(len(compute_state['embed'])),
            'true_names_f': [names_f[i] for i in y_true_f[shown]],
            'pred_names_f': pred_names_f,
            'true_names_c': [names_c[i] for i in y_true_c[shown]],
            'pred_names_c': pred_names_c,
        })


def rearrange_images():
    source_embed.data.update({
        'x':    position(0),
        'y':    position(1),
        'size': image_sizes(len(compute_state['embed'])),
    })


def draw_confusion():
    if control_state['selection_1'] is None:
        source_confusion.data.update({
            'names':     [],
            'confusion': [],
        })
        fig_confusion.y_range.factors = list(reversed(sorted(names_f)))
        fig_confusion.x_range.end = 1
    else:
        shown = compute_state['shown']
        confusion = []
        for i in range(n_classes):
            confusion.append(np.mean(compute_state['y_pred_f'][shown] == i))
        source_confusion.data.update({
            'names':     names_f,
            'confusion': confusion,
        })
        factors = [names_f[i] for i in np.argsort(confusion)]
        fig_confusion.y_range.factors = factors
        fig_confusion.x_range.end = np.min([np.max(confusion) + 0.05, 1])


def callback_selection_1(_attr, _old, new):
    new = new[0] if new else None
    if new != control_state['selection_1']:
        control_state['selection_1'] = new
        update_shown()
        update_embed()
        draw_images()
        draw_confusion()
        source_confusion.selected.indices = []


def callback_selection_2(_attr, _old, new):
    new = new[0] if new else None
    if new != control_state['selection_2']:
        control_state['selection_2'] = new
        if new is None:
            source_embed.selected.indices = []
        else:
            y_pred_shown = compute_state['y_pred_f'][compute_state['shown']]
            source_embed.selected.indices = list(
                np.where(y_pred_shown == new)[0])


def callback_image_selection(_attr, _old, new):
    y_pred_shown = compute_state['y_pred_f'][compute_state['shown']]
    correct = list(np.where(y_pred_shown == control_state['selection_2'])[0])
    if new != correct:
        source_embed.selected.indices = correct


def callback_model(_attr, old, new):
    if new != old:
        load_model(MODELS[new])
        draw_bars()
        sort_bars()
        if control_state['embed_source'] == 'Softmax':
            update_embed()
        draw_images()
        draw_confusion()


def callback_sort(_attr, old, new):
    if new != old:
        control_state['sort'] = new
        sort_bars()


def callback_embed_source(_attr, old, new):
    if new != old:
        control_state['embed_source'] = new
        update_embed()
        rearrange_images()


def callback_embed_algo(_attr, old, new):
    if new != old:
        control_state['embed_algo'] = new
        update_embed()
        rearrange_images()
        widget_recompute_tsne.visible = (new == 'TSNE')


def callback_recompute_tsne():
    update_embed()
    rearrange_images()


def callback_image_scale(_attr, old, new):
    if new != old:
        control_state['image_scale'] = new
        rearrange_images()


# Create text and control widgets
widget_text = Div(
    text='<h1>Prediction Explorer</h1>'
         '<p>A tool for exploring the predictions of image '
         'classification neural networks. Images are from the '
         'CIFAR-100 dataset.</p>'
         '<h2>Instructions</h2>'
         '<p>To see confusion bars and image embeddings, click one of '
         'the bars in the accuracy charts. To deselect, click anywhere '
         'in the empty part of the chart or press the escape key.</p>'
         '<p>To highlight the image embeddings of a specific predicted '
         'class, click one of the bars in the confusion chart.</p>'
         '<p>The image embedding area supports mouse panning and '
         'scroll wheel zooming. Click the reset tool in the top-right '
         'corner to return to the original view.</p>'
         '<h2>Controls</h2>',
    width=300)
widget_model = Select(
    title='Model',
    value='VGG16',
    options=list(MODELS.keys()),
    width=WIDGET_WIDTH)
widget_sort = Select(
    title='Sort',
    value=control_state['sort'],
    options=[
        'Alphabetical',
        'Fine Accuracy',
        'Fine Error',
        'Coarse Accuracy',
        'Coarse Error',
        'Top-5 Accuracy',
        'Top-5 Error',
    ],
    width=WIDGET_WIDTH)
widget_embed_source = Select(
    title='Embedding Source',
    value=control_state['embed_source'],
    options=[
        'Image',
        'Softmax',
    ],
    width=WIDGET_WIDTH)
widget_embed_algo = Select(
    title='Embedding Algorithm',
    value=control_state['embed_algo'],
    options=[
        'PCA',
        'Isomap',
        'Locally Linear',
        'MDS',
        'TSNE',
        'Spectral',
    ],
    width=WIDGET_WIDTH)
widget_recompute_tsne = Button(
    label='Recompute TSNE',
    width=WIDGET_WIDTH)
widget_image_scale = Slider(
    title='Image Scale',
    start=0.1, end=3,
    value=control_state['image_scale'],
    step=0.1,
    width=400)

# Set up callbacks
source_acc.selected.on_change('indices', callback_selection_1)
source_confusion.selected.on_change('indices', callback_selection_2)
source_embed.selected.on_change('indices', callback_image_selection)
widget_model.on_change('value', callback_model)
widget_sort.on_change('value', callback_sort)
widget_embed_source.on_change('value', callback_embed_source)
widget_embed_algo.on_change('value', callback_embed_algo)
widget_recompute_tsne.on_click(callback_recompute_tsne)
widget_image_scale.on_change('value', callback_image_scale)

# Initialization
load_model('vgg16')
draw_bars()
sort_bars()
update_shown()
update_embed()
draw_images()
draw_confusion()

# Create layout and show
layout = row(
    column(
        widget_text,
        widget_model,
        widget_sort,
        widget_embed_source,
        widget_embed_algo,
        widget_recompute_tsne),
    fig_acc_f,
    fig_acc_c,
    fig_acc_5,
    fig_confusion,
    Spacer(width=40),
    column(
        fig_embed,
        widget_image_scale))
curdoc().add_root(layout)
