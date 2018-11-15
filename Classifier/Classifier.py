""" MLP Classifier Base Class """

class Classifier:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    
    def fit(self, X, y):
        self.X, self.y = X, y
        return self
    
    def predict(self, X):
        pass
