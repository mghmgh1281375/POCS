import numpy as np
from sklearn.metrics import euclidean_distances

e = euclidean_distances([[1], [5], [3]], [[1], [5], [3]])
print((e))
print((e.argsort(axis=1)))
print(np.argwhere(e.argsort() == 1+0))


class BottomToUp:
    def __init__(self, X, n_clusters=5):
        self.n_clusters = n_clusters
        self.centers = X[:]
        self.clusters = X[:]

    def fit(self):
        while len(self.clusters) <= self.n_clusters:
            distances = np.argmin(euclidean_distances(self.centers, self.centers))
