""" MLP Classifier Base Class """

import numpy as np

class Classifier:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    
    def fit(self, X, y):
        self.X, self.y = X, np.array(y)
        return self
    
    def predict(self, X):
        pass
