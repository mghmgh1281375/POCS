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

# Time wrapper
def timeit(func):
    def call(*args, **kwargs):
        t0 = time()
        retval = func(*args, **kwargs)
        t1 = time()
        print(t1 - t0)
        return retval
    return call



def generator(path):
    images = os.listdir(path)
    for img in images:
        yield img, cv2.imread(os.path.join(path, img), 0)

class Tester:
    def __init__(self, data_generator, classifiers):
        """[summary]

        :param data_generator: [description]
        :type data_generator: generator
        :param classifiers: [description]
        :type classifiers: list
        """

        self.data_generator = data_generator
        self.classifiers = classifiers

    def run(self):
        for classifier in self.classifiers:
            print('Classifier {}'.format(classifier))
            # classifier.fit().predict()


if __name__ == "__main__":
    print('Test started ...')
    # print(metrics.classification_report(expected, predicted))
    # confusion_matrix = metrics.confusion_matrix(expected, predicted)
    # print(confusion_matrix)
    data_generator = generator

    Tester(data_generator, [NearestNeighbor(k=1), NearestNeighbor(k=2), Parzen(r=0.1), Bayesian()]).run()
