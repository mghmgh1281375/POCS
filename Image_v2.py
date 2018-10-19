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

class Image:
    def __init__(self, arg):

        if type(arg) is str:
            self.image = cv2.imread(arg, 0)
            cv2.imshow('original', self.image)
            cv2.waitKey()
            print(self.image.shape)
        elif type(arg) is np.ndarray:
            self.image = arg

    def normal(self, i, j):
        return self.image[int(i), int(j)]

    def average(self, i, j):
        if i > int(i):
            if j > int(j):
                pxls = [(int(i), int(j)), (int(i), int(j)+1), (int(i)+1, int(j)), (int(i)+1, int(j)+1)]
                pxls = [self.image[i, j] for i, j in pxls if i<self.image.shape[0] and j<self.image.shape[1]]
                return sum(pxls)/len(pxls)
            else:
                pxls = [(int(i), int(j)), (int(i)+1, int(j))]
                pxls = [self.image[i, j] for i, j in pxls if i<self.image.shape[0]]
                return sum(pxls)/len(pxls)
        else:
            if j > int(j):
                pxls = [(int(i), int(j)), (int(i), int(j)+1)]
                pxls = [self.image[i, j] for i, j in pxls if j<self.image.shape[1]]
                return sum(pxls)/len(pxls)
            else:
                return self.image[int(i), int(j)]
                

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
        if size[0] == -1 and size[1] != -1:
            size = int((self.image.shape[1]/self.image.shape[0])*size[1]), size[1]
        elif size[1] == -1 and size[0] != -1:
            size = int((self.image.shape[0]/self.image.shape[1])*size[0]), size[0]
        elif size == (-1, -1):
            raise ValueError('Uknown size: {}'.format(size))

        Hscale, Wscale = self.image.shape[0]/float(size[0]), self.image.shape[1]/float(size[1])
        new_image = np.ones((size), dtype='uint8')
        for i in range(size[0]):
            for j in range(size[1]):
                new_image[i, j] = method(self, i*Hscale, j*Wscale)

        return new_image

if __name__ == "__main__":
    new_image = Image('download.jpeg').resize((100, -1), Image.normal)
    cv2.imshow('1', new_image)
    new_image = Image(new_image).resize((275, 183), Image.normal)
    cv2.imshow('2', new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    