""" Bayesian Classifier """

from Classifier import Classifier
import numpy as np
from numpy import mean, std, math
from collections import defaultdict

class Bayesian(Classifier):
    def __init__(self, *args, **kwargs):
        super(Bayesian, self).__init__(*args, **kwargs)

    def fit(self, X, Y):
        self.X, self.Y = np.array(X), np.array(Y)
        self.summaries = self.__summarize_by_class__(X, Y)
        return self

    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))
 
    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

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
        
        retval = np.array([(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*X)])

        return retval

    def __summarize_by_class__(self, X, Y):
        separated = self.__separated_by_class__(X, Y)
        summaries = defaultdict(list)
        for y, x in separated.items():
            summaries[y] = self.__summarize__(x)
        
        return summaries

    def __probability__(self, x, mean_val, stdev_val):
        power_val = np.power(x-mean_val,2)
        two_power_val = (2*np.power(stdev_val,2))
        exponent = np.exp(-(power_val/two_power_val))
        retval = (1/(np.sqrt(2*np.pi)*stdev_val))*exponent
        return retval

    def predict(self, X):
        X = np.array(X)
        probabilities = {}
        for y, classSummaries in self.summaries.items():
            probabilities[y] = 1
            for i in range(len(classSummaries)):
                probabilities[y] *= self.__probability__(X[i], *classSummaries[i])
        return probabilities

if __name__ == "__main__":
    pred = Bayesian().fit([[1, 20], [2, 21], [3, 22], [4, 22]], [1, 0, 1, 0]).predict([2, 2]) or None
    print(pred)
