from sklearn import metrics

from NearestNeighbor import NearestNeighbor
from Parzen import Parzen
from Bayesian import Bayesian

# Parzen is in NearestNeighbors.radius_neightbors
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB  

from time import time

# Time wrapper
def timeit(func):
    def call(*args, **kwargs):
        t0 = time()
        retval = func(*args, **kwargs)
        t1 = time()
        print(t1 - t0)
        return retval
    return call




if __name__ == "__main__":
    print('Test started ...')
    # print(metrics.classification_report(expected, predicted))
    # confusion_matrix = metrics.confusion_matrix(expected, predicted)
    # print(confusion_matrix)
