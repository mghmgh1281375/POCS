""" One Nearest Neighbor Classifier """

from Classifier import Classifier
from Utils import Distance

class OneNearestNeighbor(Classifier):
    
    def predict(self, x):
        print(Distance.euclidean_distance(x, self.X))


if __name__ == "__main__":
    OneNearestNeighbor().fit([[1, 2], [4, 2]], [1, 2]).predict([2, 2])
