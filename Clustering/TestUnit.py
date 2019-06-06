from FeatureExtractor.Features import Intensity

from time import time

import os
import cv2
import logging
import numpy as np
import datetime

logging.basicConfig(filename='clf-{}.log'.format(datetime.datetime.now(
).strftime('%Y%m%d%H%M%S')), level=logging.INFO)

class Tester:
    def __init__(self, X, Y, X_test, Y_test, clusters):
        """[summary]

        :param X: [description]
        :type X: list
        :param clusters: [description]
        :type clusters: list
        """

        self.X, self.Y = X, Y
        self.X_test, self.Y_test = X_test, Y_test
        self.clusters = clusters

    def run(self):
        for cluster in self.clusters:
            success = 0
            clf = cluster.fit(self.X, 40000)
            for i, x in enumerate(self.X_test):
                predicted = clf.predict(x)
                print('Actual: {}, Predicted: {}'.format(self.Y_test[i], predicted))
                # if predicted:
                #     if predicted == self.Y_test[i]:
                #         success += 1
            del(cluster)


if __name__ == "__main__":
    logging.info('Test started ...')
    # print(metrics.classification_report(expected, predicted))
    # confusion_matrix = metrics.confusion_matrix(expected, predicted)
    # print(confusion_matrix)
    # data_generator = generator

    # Tester(data_generator, [NearestNeighbor(k=1), NearestNeighbor(k=2), Parzen(r=0.1), Bayesian()]).run()

    from pickle import dump, load
    from SOM import SOM
    clusters = [SOM(10, (6, 6))]
    intensity = Intensity()
    train_X = load(open('MLP/resources/train-100d-X.pickle', 'rb'))
    train_Y = load(open('MLP/resources/train-100d-Y.pickle', 'rb'))
    test_X = load(open('MLP/resources/test-100d-X.pickle', 'rb'))
    test_Y = load(open('MLP/resources/test-100d-Y.pickle', 'rb'))

    Tester(train_X, train_Y, test_X, test_Y, clusters=clusters).run()

