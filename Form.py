from time import time

import cv2
import numpy as np

from Preprocess import Preprocess

def show(title, img):
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey()

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
    
    def __del__(self):
        cv2.destroyAllWindows()

    def preprocess(self):
        self.prep_obj = Preprocess(self.image, square_size = 57, vertical_crop=(600, 3000))
        self.prep_obj.find_rows().find_cols_boundries()
        return self

    def segment_row(self, row):
        """returns row segments
        First segment is handwrite text,
        Others are digits.
        
        Arguments:
            row {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return [row[:, 260:930], row[:, 970:1076], row[:, 1080:1186], row[:, 1193:1300], row[:, 1340:1444], row[:, 1450:1550]]

    def get_rows(self, margin=65):
        for row in self.prep_obj.rows:
            row_img = self.image[
                int(self.prep_obj.kwargs['vertical_crop'][0] + row - margin):
                int(self.prep_obj.kwargs['vertical_crop'][0] + row + margin)]
            show('row', row_img)
            show('segment', np.hstack(self.segment_row(row_img)))

if __name__ == '__main__':
    Form('/home/mohammad/Desktop/Forms/form1/00000010.jpg').preprocess().get_rows()

