""" Parzen Classifier """

from Classifier import Classifier

class Parzen(Classifier):
    def __init__(self, r, *args, **kwargs):
        super(Parzen, self).__init__(*args, **kwargs)
        self.r = r

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass

