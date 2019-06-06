#!/usr/bin/env python
# coding: utf-8

"""This module is for training cnn on Hoda dataset
"""

# This codes are for using https://colab.research.google.com
# get_ipython().system(' rm -r class.vision')
# get_ipython().system(' git clone https://github.com/Alireza-Akhavan/class.vision')
# get_ipython().system(' ls')
# get_ipython().system(' mv class.vision/* ./')
# get_ipython().system(' ls')

from ModelCNN import ModelCNN

import keras
import numpy as np

from dataset import load_hoda

np.random.seed(123)  # for reproducibility

x_train_original, y_train_original, x_test_original, y_test_original = load_hoda(training_sample_size=58000, test_sample_size=2000, size=20)

def print_data_info(x_train, y_train, x_test, y_test):
    #Check data Type
    print ("\ttype(x_train): {}".format(type(x_train)))
    print ("\ttype(y_train): {}".format(type(y_train)))

    #check data Shape
    print ("\tx_train.shape: {}".format(np.shape(x_train)))
    print ("\ty_train.shape: {}".format(np.shape(y_train)))
    print ("\tx_test.shape: {}".format(np.shape(x_test)))
    print ("\ty_test.shape: {}".format(np.shape(y_test)))

    #sample data
    print ("\ty_train[0]: {}".format(y_train[0]))


# Preprocess input data for Keras. 
x_train = np.array(x_train_original)
y_train = keras.utils.to_categorical(y_train_original, num_classes=10)
x_test = np.array(x_test_original)
y_test = keras.utils.to_categorical(y_test_original, num_classes=10)

print("Before Preprocessing:")
print_data_info(x_train_original, y_train_original, x_test_original, y_test_original)
print("After Preprocessing:")
print_data_info(x_train, y_train, x_test, y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.reshape(x_train, (-1, 20, 20, 1))
x_test = np.reshape(x_test, (-1, 20, 20, 1))
print(x_train.shape)

model = ModelCNN()
model.fit(x_train, y_train,
          epochs=64,
          batch_size=512,
          ) # validation_split=0.2


loss, acc = model.evaluate(x_test, y_test)
print('\nTesting loss: %.2f, acc: %.2f%%'%(loss, acc))
# Output: Testing loss: 0.02, acc: 0.99%


# Saving model from https://colab.research.google.com
# from datetime import datetime
# name = 'model_conv_{}.h5'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
# model.save(name)
# from google.colab import files
# files.download(name)


# Wrong predictions
predicted_classes = model.predict_classes(x_test)
idxs = np.argwhere(predicted_classes != y_test_original)
idxs = np.reshape(idxs, (idxs.shape[0],))
import matplotlib.pyplot as plt
plt.imshow(np.vstack([np.reshape(x_test[idx], (20, 20)) for idx in idxs]))
