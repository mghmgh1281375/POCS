from time import time

import cv2
import numpy as np

from Preprocess import Preprocess


def timeit(func):
    def call(*args, **kwargs):
        t0 = time()
        ret = func(*args, **kwargs)
        t1 = time()
        print(t1 - t0)
        return ret
    return call

class Form:
    def __init__(self, arg):
        """[summary]
        
        Arguments:
            arg {path or np.ndarray} -- if passing np.ndarray, image must be gray mode (one channel)
        """

        if type(arg) is str:
            self.image = cv2.imread(arg, 0)
        elif type(arg) is np.ndarray:
            self.image = arg

    def preprocess(self):
        prep_obj = Preprocess(self.image, SQUARE_SIZE = 57)
        prep_obj.find_rows().find_cols_boundries()
        print(prep_obj.rows)
        print(prep_obj.cols_boundries)



if __name__ == '__main__':
    Form('/home/mohammad/Desktop/Forms/form1/00000010.jpg').preprocess()
