""" Features """
import numpy as np

class Intensity:
    def __init__(self):
        print('Intensity instance initialized.')
    
    def extract(self, matrix):
        X = np.sum(matrix, axis=0)
        Y = np.sum(matrix, axis=1)
        return np.concatenate((X, Y))

if __name__ == "__main__":
    # Intensity
    features = Intensity().extract([[1, 1], [3, 3]])
    print(features)

