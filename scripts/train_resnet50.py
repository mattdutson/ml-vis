#!/usr/bin/env python3

import os.path as path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import *
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# Define training parameters
name = 'resnet50'
learning_rate = 1e-5
epochs = 100

# Load and preprocess training data
(x_train, y_train), _ = cifar100.load_data(label_mode='fine')
y_train = tf.one_hot(tf.squeeze(y_train), 100)

# Load pretrained backbone
backbone = ResNet50(input_shape=x_train.shape[1:],
                    include_top=False,
                    weights='imagenet')

# Add final size-specific layers
inputs = backbone.input
x = inputs
x = backbone(x)
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
x = Softmax()(x)
model = Model(inputs=inputs, outputs=x)

# Compile the model for training
model.compile(optimizer=RMSprop(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# Set up callbacks
tb_name = '{}_{}'.format(name, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
callbacks = [
    ModelCheckpoint(path.join('models', name + '.h5'),
                    monitor='val_accuracy',
                    verbose=1,
                    save_best_only=True),
    TensorBoard(log_dir=path.join('tensorboard', tb_name)),
]

# Train the model (best epoch automatically saved by the callback)
model.fit(x=x_train,
          y=y_train,
          epochs=epochs,
          callbacks=callbacks,
          validation_split=0.2)
