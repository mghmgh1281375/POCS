""" Bayesian Classifier """

from Classifier import Classifier
import numpy as np
from numpy import mean, std
import math
from collections import defaultdict

class Bayesian(Classifier):
    def __init__(self, *args, **kwargs):
        super(Bayesian, self).__init__(*args, **kwargs)

    def __separated_by_class__(self, X, Y):
        separated = defaultdict(list)
        for i in range(len(X)):
            separated[Y[i]].append(X[i])
        return separated
        
    def __summarize__(self, X):
        """Summarize one class data
        
        :param X: [description]
        :type X: [type]
        """
        return [(mean(attribute), std(attribute)) for attribute in zip(*X)]
        
    def __summarize_by_class__(self, X, Y):
        separated = self.__separated_by_class__(X, Y)
        summaries = defaultdict(list)
        for y, x in separated.items():
            summaries[y] = self.__summarize__(x)
        return summaries
    
    def __probability__(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1/(math.sqrt(2*math.pi)*stdev))*exponent     


    def fit(self, X, Y):
        self.X, self.Y = X, Y
        self.summaries = self.__summarize_by_class__(X, Y)



    
    def predict(self, X):
        probabilities = {}
        for y, classSummaries in self.summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                probabilities[y] *= calculateProbability(X[i], *classSummaries[i])
        return probabilities

if __name__ == "__main__":
    pred = Bayesian().fit([[1, 2], [4, 2]], [1, 2]).predict([[2, 2]]) or None
    print(pred)
