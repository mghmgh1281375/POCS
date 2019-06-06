#!/usr/bin/env python
# coding: utf-8

"""This module is for predicting input images with size (20, 20)"""


from ModelCNN import ModelCNN

import keras.utils
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_hoda

np.random.seed(123)  # for reproducibility

x_train_original, y_train_original, x_test_original, y_test_original = load_hoda(training_sample_size=58000, test_sample_size=2000, size=20)

x_test = np.array(x_test_original)
y_test = keras.utils.to_categorical(y_test_original, num_classes=10)

x_test = x_test.astype('float32')
x_test /= 255

x_test = np.reshape(x_test, (-1, 20, 20, 1))
print(x_test[0].shape)
print(x_test[0][0])
print(x_test.shape)
plt.imshow(np.reshape(x_test[0], (20, 20)))
plt.show()
# model = ModelCNN()
# model.load_weights('models/model_conv_20181024081423.h5')
# classes, probas = model.predict_batch(x_test)
# print('Classes', classes)
# print('Probas', probas)
