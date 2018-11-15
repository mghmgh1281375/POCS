""" Utils module """

from math import sqrt
import numpy as np

class Distance:

    @staticmethod
    def euclidean_distance(x, X):
        """measure distance between `x` and every elements in `X`
        
        Arguments:
            x {list} -- 1 dim array
            X {matrix} -- 2 dim array - array of feature vectors
        """
        X = np.array(X)
        distances = []
        for i in range(len(X)):
            distances.append(np.sum(np.sqrt(np.power(X[i]-x, 2))))

        distances = np.array(distances)
        print(distances.shape)
        return distances
        

