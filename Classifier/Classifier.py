""" MLP Classifier Base Class """
from abc import ABCMeta, abstractmethod
import numpy as np

class Classifier(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    
    def fit(self, X, Y):
        self.X, self.Y = X, Y
        return self
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def __del__(self):
        self.X, self.Y = None, None

    def __repr__(self):
        return str(self.__class__) + ' - ' + str(self.args) + ' - ' + str(self.kwargs)
