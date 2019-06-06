from sklearn import metrics

from FeatureExtractor.Features import Intensity

from time import time
from MLP.mlp import Mlp
# from functools import partial
import numpy as np
import os
import cv2
import datetime
import logging
from pickle import dump, load


logging.basicConfig(filename='mlp-{}.log'.format(datetime.datetime.now(
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
    logging.info('Loading path=\'{}\' {} images with size=({}, {})'.format(
        path, len(images), size, size))
    for img in images:
        try:
            image = cv2.imread(os.path.join(path, img), 0)
            dilated = cv2.dilate(image, np.ones(
                (3, 3), np.uint8), iterations=1)
            _, image = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)
            image = focus(image)
            image = cv2.resize(image, (size, size))
            features = feature_extractor(image)
            data.append(features)
        except Exception as e:
            logging.warning('{}, IMAGE_PATH={}'.format(
                str(e), os.path.join(path, img)))

    retval = np.array(data), np.array(
        [int(img.split('-')[0]) for img in images])
    # logging.info('retval shape: {} {}'.format(retval[0].shape, retval[0].shape))
    # print(retval[0].shape, retval[0].shape)
    return retval


class Tester:
    def __init__(self, X, Y, X_test, Y_test, classifiers, iterations):
        """[summary]

        :param X: [description]
        :type X: list
        :param classifiers: [description]
        :type classifiers: list
        """

        self.X, self.Y = X, Y
        self.X_test, self.Y_test = X_test, Y_test
        self.classifiers = classifiers
        self.iterations = iterations

    def run(self):
        for classifier in self.classifiers:
            success = 0
            logging.info('Classifier {}'.format(classifier))
            clf = classifier.fit(self.X, self.Y, iterations=self.iterations)
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

    train_path = '/home/mohammad/Downloads/POCS-persian-number-dataset/train'
    test_path = '/home/mohammad/Downloads/POCS-persian-number-dataset/test'

    intensity = Intensity()

    # Saving to pickle
    # X, Y = load_data(train_path, intensity.extract, 50)
    # dump(X, open('MLP/resources/train-X.pickle', 'wb'))
    # dump(Y, open('MLP/resources/train-Y.pickle', 'wb'))

    # X, Y = load_data(test_path, intensity.extract, 50)
    # dump(X, open('MLP/resources/test-X.pickle', 'wb'))
    # dump(Y, open('MLP/resources/test-Y.pickle', 'wb'))

    classifiers = [Mlp([100, 64, 64, 16, 12])]
    # for size in range(10, 200, 10):
    # Tester(*load_data(train_path, intensity.extract, size), *load_data(test_path,
    #                                                             intensity.extract, size), classifiers=classifiers, iterations=1200).run()

    train_X = load(open('MLP/resources/train-100d-X.pickle', 'rb'))

    from sklearn import preprocessing
    lbl_bin = preprocessing.LabelBinarizer()

    train_Y = load(open('MLP/resources/train-100d-Y.pickle', 'rb'))
    train_Y = lbl_bin.fit_transform(train_Y)

    test_X = load(open('MLP/resources/test-100d-X.pickle', 'rb'))
    test_Y = load(open('MLP/resources/test-100d-Y.pickle', 'rb'))
    test_Y = lbl_bin.transform(test_Y)


    Tester(train_X, train_Y, test_X, test_Y, classifiers=classifiers, iterations=1200).run()
