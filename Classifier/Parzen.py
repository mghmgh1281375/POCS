""" Parzen Classifier """

from Classifier import Classifier
from Utils import Distance
import numpy as np

class Parzen(Classifier):
    def __init__(self, r, *args, **kwargs):
        super(Parzen, self).__init__(*args, **kwargs)
        self.r = r
    
    def predict(self, X):
        ret = []
        for x in X:
            ret.append(self.y[np.where(Distance().euclidean_distance(x, self.X) <= self.r)])
        return np.array(ret)

if __name__ == "__main__":
    pred = Parzen(r=1).fit([[1, 2], [4, 2]], [1, 2]).predict([[2, 2]])
    print(pred)

