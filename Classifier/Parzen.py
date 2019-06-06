""" Parzen Classifier """

# TODO: Needs to select best radius

from Classifier import Classifier
from Utils import Distance
import numpy as np
from collections import Counter

class Parzen(Classifier):
    def __init__(self, r=0.1, *args, **kwargs):
        super(Parzen, self).__init__(r, *args, **kwargs)
        self.r = r
        # TODO: Selecting best radius
    
    def predict(self, X):
        ret = self.Y[np.where(Distance().euclidean_distance(X, self.X) <= self.r)]
        if len(ret):
            return [max(Counter(ret))]

if __name__ == "__main__":
    X = np.array([[1, 2, 3, 2, 2], [4, 2, 5, 6, 7], [4, 2, 4, 6, 9]])
    Y = np.array([1, 2, 2])
    pred = Parzen(r=20).fit(X, Y).predict([2, 2, 2, 2, 2])
    print(pred)


    from sklearn.neighbors import NearestNeighbors
    # t2 = time()
    sk = NearestNeighbors()
    sk.fit(X, Y)
    # for r in range(0, 10000, 100):
    pred = sk.radius_neighbors([[2, 2, 2, 2, 2]], radius=10)
    # t3 = time()
    print(pred)
    # print('scikit-learn\'s NearestNeighbors: ({})'.format(t3 - t2), pred)
