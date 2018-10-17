"""  """

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

    @timeit
    def resize(self, size):
        self.old_height, self.old_width = self.image.shape
        self.new_height, self.new_width = size
        self.kernel_height, self.kernel_width = (self.old_height - self.new_height) + 1, (self.old_width - self.new_width) + 1
        print(self.kernel_height, self.kernel_width)
        kernel = np.ones((self.kernel_height, self.kernel_width))
        return self.convolve(self.image, kernel)

    def convolve(self, image, kernel):
        new_image = [[0] * self.new_width for i in range(self.new_height)]
        for i in range(image.shape[0] - kernel.shape[0] + 1):
            for j in range(image.shape[1] - kernel.shape[1] + 1):
                p = image[i: i + kernel.shape[0], j: j + kernel.shape[1]]
                new_image[i][j] = np.sum(kernel*p)/(kernel.shape[0]*kernel.shape[1])
        return np.array(new_image, dtype='uint8')

if __name__ == "__main__":
    new_image = Image('download.jpeg').resize((50, 150))
    cv2.imshow('resized (150, 85)', new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    