""" One Nearest Neighbor Classifier """

from Classifier import Classifier
from Utils import Distance
import numpy as np

class NearestNeighbor(Classifier):
    def __init__(self, k=1, *args, **kwargs):
        super(NearestNeighbor, self).__init__(*args, **kwargs)
    
    def predict(self, X):
        # TODO: Incomplete
        ret = []
        for x in X:
            ret = self.y[np.argmin(Distance.euclidean_distance(x, self.X), axis=0)]
        return np.array(ret)


if __name__ == "__main__":
    pred = NearestNeighbor(k=1).fit([[1, 2], [4, 2]], [1, 2]).predict([[2, 2]])
    print(pred)
