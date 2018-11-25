""" Bayesian Classifier """

from Classifier import Classifier
import numpy as np

class Bayesian(Classifier):
    def __init__(self, *args, **kwargs):
        super(Bayesian, self).__init__(*args, **kwargs)
    
    def predict(self, X):
        preds = []
        for x in X:
            g = None
            stdev = np.std(x)
            mean = np.mean(x)
            g = -X.shape[1]/2 * np.log2(2 * np.pi) - 0.5 * np.log2() - 0.5 * (x - mean) 
            preds.append(g)

if __name__ == "__main__":
    pred = Bayesian().fit([[1, 2], [4, 2]], [1, 2]).predict([[2, 2]])
    print(pred)

