""" Features """
import numpy as np

class Intensity:
    def __init__(self):
        print('Intensity instance initialized.')
    
    def extract(self, matrix, normalization=True):
        """I recommending use binary inversed image
        
        :param matrix: [description]
        :type matrix: [type]
        :param normalization: [description], defaults to True
        :param normalization: bool, optional
        :return: [description]
        :rtype: [type]
        """

        X = np.sum(matrix, axis=0)
        Y = np.sum(matrix, axis=1)

        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)

        if normalization:
            X = X/max(X)
            Y = Y/max(Y)
        return np.concatenate((X, Y))

if __name__ == "__main__":
    # Intensity
    features = Intensity().extract([[1, 1], [3, 3]])
    print(features)

