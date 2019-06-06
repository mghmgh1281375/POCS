""" Utils module """

from math import sqrt
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class Distance:

    @staticmethod
    def euclidean_distance(x, X):
        """measure distance between `x` and every elements in `X`
        
        Arguments:
            x {list} -- 1 dim array
            X {matrix} -- 2 dim array - array of feature vectors
        """
        
        distances = []
        for i in range(len(X)):
            distances.append(np.sum(np.sqrt(np.power(X[i]-x, 2))))
            # distances.append(float(cosine_similarity([x], [X[i]])))

        distances = np.array(distances)
        # print(distances.shape)
        return distances
