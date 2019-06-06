import numpy as np


class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_idx, data_point in enumerate(X):
            for center_idx, center in enumerate(self.centers):
                G[data_point_idx, center_idx] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

if __name__ == "__main__":

    from pickle import dump, load

    train_X = load(open('MLP/resources/train-100d-X.pickle', 'rb'))
    train_Y = load(open('MLP/resources/train-100d-Y.pickle', 'rb'))
    test_X = load(open('MLP/resources/test-100d-X.pickle', 'rb'))
    test_Y = load(open('MLP/resources/test-100d-Y.pickle', 'rb'))
    print('train_X before cleaning', train_X.shape)
    
    # print(np.isnan(train_X[:900]))
    train_X = train_X[:900]
    train_Y = train_Y[:900]
    # print(np.isnan(test_X[:900]))
    test_X = test_X[:900]
    test_Y = test_Y[:900]



    print('train_X after cleaning', train_X.shape)

    rbfn = RBFN(hidden_shape=100)
    rbfn.fit(train_X, train_Y)
    for i in range(300):
        print(rbfn.predict([test_X[i]]), test_Y[i])
