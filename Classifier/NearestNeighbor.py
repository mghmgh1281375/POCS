""" One Nearest Neighbor Classifier """

from Classifier import Classifier
from Utils import Distance
import numpy as np
from collections import Counter

class NearestNeighbor(Classifier):
    def __init__(self, k=1, *args, **kwargs):
        super(NearestNeighbor, self).__init__(k=1, *args, **kwargs)
        self.k = k
    
    def predict(self, x):
        # TODO: Incomplete
        ret = self.Y[np.argsort(Distance.euclidean_distance(x, self.X), axis=0)[:self.k]] 
        return np.array([max(Counter(ret))])

if __name__ == "__main__":
    from time import time
    t0 = time()
    pred = NearestNeighbor(k=3).fit(np.array([[1, 2], [4, 2], [2, 3], [1, 5],]), np.array([1, 2, 2, 1])).predict(np.array([2, 2]))
    t1 = time()
    print('My NearestNeighbors: ({})'.format(t1 - t0), pred)

    # from sklearn.neighbors import NearestNeighbors
    # t2 = time()
    # sk = NearestNeighbors()
    # sk.fit([[1, 2], [4, 2]], [1, 2])
    # pred = sk.kneighbors([[2, 2]], n_neighbors=2)[0]
    # t3 = time()
    # print('scikit-learn\'s NearestNeighbors: ({})'.format(t3 - t2), pred)
