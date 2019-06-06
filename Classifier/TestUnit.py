from sklearn import metrics

from NearestNeighbor import NearestNeighbor
from Parzen import Parzen
from Bayesian import Bayesian

from FeatureExtractor.Features import Intensity

# Parzen is in NearestNeighbors.radius_neightbors
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB

from time import time

# For using 1-NearestNeighbors and K-NearestNeighbors
from functools import partial
import os
import cv2
import logging
import numpy as np
import datetime

logging.basicConfig(filename='clf-{}.log'.format(datetime.datetime.now(
).strftime('%Y%m%d%H%M%S')), level=logging.INFO)


# Time wrapper
def timeit(func):
    def call(*args, **kwargs):
        t0 = time()
        retval = func(*args, **kwargs)
        t1 = time()
        print(t1 - t0)
        return retval
    return call

def focus(img):
    """Removing white area around 
    
    :param img: input one channel image (grayscale)
    :type img: numpy.ndarray
    :return: focused one channel image (grayscale)
    :rtype: numpy.ndarray
    """


    X = np.sum(img, axis=0)
    Y = np.sum(img, axis=1)
    x_min, x_max, y_min, y_max = None, None, None, None

    for i in range(X.shape[0]):
        if X[i] != 0 and x_min is None:
            x_min = i
        if X[-i] != 0 and x_max is None:
            x_max = -i
    for i in range(Y.shape[0]):
        if Y[i] != 0 and y_min is None:
            y_min = i
        if Y[-i] != 0 and y_max is None:
            y_max = -i

    return img[y_min:y_max, x_min:x_max]

def load_data(path, feature_extractor, size):
    data = []
    images = os.listdir(path)
    logging.info('Loading path=\'{}\' {} images with size=({}, {})'.format(path, len(images), size, size))
    for img in images:
        try:
            image = cv2.imread(os.path.join(path, img), 0)
            dilated = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
            _, image = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)
            image = focus(image)
            image = cv2.resize(image, (size, size))
            features = feature_extractor(image)
            data.append(features)
        except Exception as e:
            logging.warning('{}, IMAGE_PATH={}'.format(str(e), os.path.join(path, img)))

    return np.array(data), np.array([int(img.split('-')[0]) for img in images])


class Tester:
    def __init__(self, X, Y, X_test, Y_test, classifiers):
        """[summary]

        :param X: [description]
        :type X: list
        :param classifiers: [description]
        :type classifiers: list
        """

        self.X, self.Y = X, Y
        self.X_test, self.Y_test = X_test, Y_test
        self.classifiers = classifiers

    def run(self):
        for classifier in self.classifiers:
            success = 0
            logging.info('Classifier {}'.format(classifier))
            clf = classifier.fit(self.X, self.Y)
            logging.info('Predicting')
            for i, x in enumerate(self.X_test):
                predicted = clf.predict(x)
                logging.debug('pred({}) == target({})'.format(
                    predicted, self.Y_test[i]))
                if predicted:
                    if predicted == self.Y_test[i]:
                        success += 1
            logging.info('Classifier {}: Accuracy: {}'.format(
                classifier, float(success)/self.Y.shape[0]))
            del(classifier)


if __name__ == "__main__":
    logging.info('Test started ...')
    # print(metrics.classification_report(expected, predicted))
    # confusion_matrix = metrics.confusion_matrix(expected, predicted)
    # print(confusion_matrix)
    # data_generator = generator

    # Tester(data_generator, [NearestNeighbor(k=1), NearestNeighbor(k=2), Parzen(r=0.1), Bayesian()]).run()
    train_path = '/home/mohammad/Downloads/POCS-persian-number-dataset/train'
    test_path = '/home/mohammad/Downloads/POCS-persian-number-dataset/test'
    classifiers = [NearestNeighbor(k=1), NearestNeighbor(
        k=3), Parzen(r=0.1), Bayesian()]
    # classifiers = [Parzen(r=100000)]
    intensity = Intensity()
    for size in range(10, 200, 10):
        Tester(*load_data(train_path, intensity.extract, size), *load_data(test_path,
                                                                    intensity.extract, size), classifiers=classifiers).run()

