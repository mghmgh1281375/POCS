#!/usr/bin/env python
# coding: utf-8

"""This module is for predicting input images with size (20, 20)"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D
import numpy as np

class ModelCNN:
    def __init__(self, *args, **kwargs):
        self.model = Sequential()
        self.model.add(Conv2D(32, (10, 10), activation='relu', input_shape=(20, 20, 1)))
        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
            
    def load_weights(self, path):
        self.model.load_weights(path)

    def predict_single(self, image):
        class_ = self.model.predict_classes(np.expand_dims(image, axis=0))[0]
        proba_ = self.model.predict_proba(np.expand_dims(image, axis=0))[0]
        return class_, proba_
    
    def predict_batch(self, images):
        classes_ = self.model.predict_classes(images)
        probas_ = self.model.predict_proba(images)
        return (classes_, probas_)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict_classes(self, *args, **kwargs):
        return self.model.predict_classes(*args, **kwargs)
