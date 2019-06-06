from sklearn.metrics import euclidean_distances
import numpy as np
import scipy.stats as st
from tqdm import tqdm
import logging

root = logging.getLogger()
handler = logging.StreamHandler()
bf = logging.Formatter('{asctime} {name} {levelname:8s} {message}',
                       style='{')
handler.setFormatter(bf)
root.addHandler(handler)
logger = logging.getLogger('foo.bar')


class SOM(object):
    def __init__(self, input_size: int, output_size: int, learning_rate=0.001):
        """[summary]
        
        :param input_size: input size
        :type input_size: int
        :param output_size: output size
        :type output_size: int
        :param learning_rate: update weight learning rate, defaults to 0.001
        :param learning_rate: float, optional
        """

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.output_weights = np.random.uniform(
            low=0, high=1, size=(output_size[0], output_size[1], input_size))
        self.gaussian_kernel = self._gaussian_kernel(max(output_size)*2)

    def _neighbourhood(self, winner_pos, neighbour_pos) -> float:
        """gaussian neighbourhood function
        
        :param winner_pos: position of winner hidden state
        :type winner_pos: tuple
        :param neighbour_pos: position of neighbour
        :type neighbour_pos: tuple
        :return: returns neighbourhood function for given position
        :rtype: float
        """

        pos = np.absolute(np.array(self.output_size)/2) + \
            np.array(neighbour_pos) - np.array(winner_pos)
        return self.gaussian_kernel[int(pos[0]), int(pos[1])]

    def _gaussian_kernel(self, size: int, nsig=3) -> np.ndarray:
        """generate gaussian kernel (matrix)
        
        :param size: shape of kernel
        :type size: int
        :param nsig: [description], defaults to 3
        :param nsig: int, optional
        :return: generated kernel
        :rtype: np.ndarray
        """

        interval = (2*nsig+1.)/(size)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., size+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        return kernel_raw/kernel_raw.sum()

    def _update(self, x):
        """update weights with respect to single input
        
        :param x: single input
        :type x: list | np.ndarray
        """

        minimum_distance, minimum_pos = float('inf'), (-1, -1)
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                dist = np.squeeze(euclidean_distances(
                    [x], [self.output_weights[i][j]]))
                if dist < minimum_distance:
                    minimum_distance, minimum_pos = dist, (i, j)

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                self.output_weights[i][j] += self.learning_rate * \
                    self._neighbourhood(minimum_pos, (i, j)) * minimum_distance

    def fit(self, X, n_iterations):
        """fitting network with given list | np.ndarray of items
        
        :param X: list of items
        :type X: list | np.ndarray
        :param n_iterations: number of iterations
        :type n_iterations: int
        :return: return class instance
        :rtype: SOM
        """

        length = len(X)
        for iteration in tqdm(range(n_iterations)):
            self._update(X[iteration % length])
        return self

    def predict(self, x: np.ndarray):
        """returns position (i, j) for single input
        
        :param x: single input list | np.ndarray
        :type x: list | np.ndarray
        :return: returns tuple of position and similarity of winner.
        :rtype: tuple
        """

        minimum_distance, minimum_pos = float('inf'), (-1, -1)
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                dist = np.squeeze(euclidean_distances(
                    [x], [self.output_weights[i][j]]))
                if dist < minimum_distance:
                    minimum_distance, minimum_pos = dist, (i, j)
        return minimum_pos, minimum_distance


if __name__ == "__main__":
    from pickle import dump, load

    train_X = load(open('MLP/resources/train-100d-X.pickle', 'rb'))
    train_Y = load(open('MLP/resources/train-100d-Y.pickle', 'rb'))
    test_X = load(open('MLP/resources/test-100d-X.pickle', 'rb'))
    test_Y = load(open('MLP/resources/test-100d-Y.pickle', 'rb'))
    print('train_X before cleaning', train_X.shape)

    dimX = train_X.shape[1]

    train_X = train_X[~np.isnan(train_X)].reshape(-1, 100)
    train_Y = train_Y[~np.isnan(train_Y)]
    test_X = test_X[~np.isnan(test_X)].reshape(-1, 100)
    test_Y = test_Y[~np.isnan(test_Y)]

    print('train_X after cleaning', train_X.shape)

    som = SOM(100, (6, 6)).fit(train_X, 10000)
    for i in range(300):
        print(som.predict(train_X[i]), train_Y[i])
