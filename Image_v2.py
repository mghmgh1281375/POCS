""" Image module version 2 """

import numpy as np
import cv2
from time import time

def timeit(func):
    def call(*args, **kwargs):
        t0 = time()
        ret = func(*args, **kwargs)
        t1 = time()
        print(t1 - t0)
        return ret
    return call


class Filter:
    def __init__(self, size):
        pass

class Image:
    def __init__(self, path):
        self.image = cv2.imread(path, 0)
        cv2.imshow('original', self.image)
        cv2.waitKey()
        print(self.image.shape)

    def normal(self, i, j):
        return self.image[i, j]

    def average(self, i, j):
        pass

    def minimum(self, i, j):
        pass

    def maximum(self, i, j):
        pass
    
    def median(self, i, j):
        pass

    def first(self, i, j):
        pass

    @timeit
    def resize(self, size, method):
        Hscale, Wscale = self.image.shape[0]/float(size[0]), self.image.shape[1]/float(size[1])
        new_image = np.ones((size), dtype='uint8')
        for i in range(size[0]):
            for j in range(size[1]):
                new_image[i, j] = self.image[int(i*Hscale), int(j*Wscale)]

        # if method == 'min':
        #     method = self.minimum
        # elif method == 'max':
        #     method = self.maximum
        # elif method == 'med':
        #     method = self.median
        # elif method == 'avg':
        #     method = self.average
        # elif method == 'first':
        #     method = self.first

        return new_image

if __name__ == "__main__":
    new_image = Image('download.jpeg').resize((275, 100), 'avg')
    cv2.imshow('resized (150, 85)', new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    