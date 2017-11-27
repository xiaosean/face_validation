"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import pandas
import resnet
import os


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet64_faceValidation.csv')
directory = "./lfw_only_face_resize/"
classes = os.listdir(directory)
# out_directory = "./lfw_only_face_resize/"
batch_size = 32
nb_classes = classes
nb_epoch = 100000
data_augmentation = True

# input image dimensions
img_rows, img_cols = 104, 96
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# X_test -= mean_image
# X_train /= 128.
# X_test /= 128.




model = resnet.ResnetBuilder.build_resnet_64((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True,
          verbose=1,
          callbacks=[lr_reducer, early_stopper, csv_logger])
