from scipy import random, exp, zeros, dot
from scipy.linalg import norm, pinv
import numpy as np

from matplotlib import pyplot as plt


class RBF:
    def __init__(self, indim, numCenters, outdim):
        """Radial Basis Funciton

        :param indim: input dimension
        :type indim: int
        :param numCenters: hidden dimension
        :type numCenters: int
        :param outdim: output dimension
        :type outdim: int
        """

        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim)
                        for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        """[summary]

        :param c: [description]
        :type c: [type]
        :param d: [description]
        :type d: [type]
        :return: [description]
        :rtype: [type]
        """

        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
        """calculate activations of RBF's

        :param X: [description]
        :type X: [type]
        :return: [description]
        :rtype: [type]
        """

        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def fit(self, X, Y):
        """fitting network with given X, Y

        :param X: matrix of dimensions n x indim
        :type X: [type]
        :param Y: column vector of dimension n x 1
        :type Y: [type]
        """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        # calculate activations of RBFs
        G = self._calcAct(X)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def predict(self, X):
        """return of y for given X

        :param X: matrix of dimensions n x indim
        :type X: list
        :return: [description]
        :rtype: [type]
        """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y


if __name__ == "__main__":

    from pickle import dump, load

    train_X = load(open('MLP/resources/train-100d-X.pickle', 'rb'))
    train_Y = load(open('MLP/resources/train-100d-Y.pickle', 'rb'))
    test_X = load(open('MLP/resources/test-100d-X.pickle', 'rb'))
    test_Y = load(open('MLP/resources/test-100d-Y.pickle', 'rb'))
    print('train_X before cleaning', train_X.shape)

    mask = np.any(np.isnan(train_X), axis=1)
    train_X = train_X[mask]
    train_Y = train_Y[mask]
    mask = np.any(np.isnan(test_X), axis=1)
    test_X = test_X[mask]
    test_Y = test_Y[mask]

    print('train_X after cleaning', train_X.shape)

    # rbfn = RBF(100, 120, 12)
    # rbfn.fit(train_X, train_Y)
    # for i in range(300):
    #     print(rbfn.predict([test_X[i]]), test_Y[i])
