
import numpy as np
import matplotlib.pyplot as plt

def plot(input):
    plt.plot(input)
    plt.show()

def unified(input): return 1 if input > 0 else 0

def mean_squared_error(y, y_): return np.mean(np.power(y - y_, 2))

mse = mean_squared_error

class Perceptron():

    def __init__(self, n_input, epochs, learning_rate=0.01, initializer=np.zeros, activation=unified, loss=mse):
        self.n_input = n_input
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.weights = initializer((n_input + 1))
        self.activation = activation
        self.loss = loss

    def train(self, X, Y, callback=None):
        losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            for x, y in zip(X, Y):
                predict = self.predict(x)
                loss = self.loss(predict, y)
                epoch_losses.append(loss)
                # update weights
                self.weights[1:] += (self.learning_rate * (y - predict)) * x
                # update bias
                self.weights[0] += self.learning_rate * (y - predict)
            losses.append(np.mean(epoch_losses))
        callback(losses)

    def predict(self, X):
        a = np.dot(X, self.weights[1:]) + self.weights[0]
        return self.activation(a)

if __name__ == "__main__":

    perceptron = Perceptron(2, 100, 0.01, np.random.rand)

    data = {
        'and': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'Y': [0, 0, 0, 1]
        },
        'xor': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'Y': [0, 1, 1, 0]
        }
    }

    perceptron.train(data['and']['X'], data['and']['Y'], callback=plot)
    print(perceptron.weights)
    print(perceptron.predict([0, 0]))
    print(perceptron.predict([0, 1]))
    print(perceptron.predict([1, 0]))
    print(perceptron.predict([1, 1]))
