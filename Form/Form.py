from time import time

import cv2
import numpy as np

from Preprocess import Preprocess
from ModelCNN import ModelCNN


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

model = ModelCNN()
# model.load_weights('/home/mohammad/Projects/POCS/Form/models/model_conv_20181024081423.h5')
# classes, probas = model.predict_batch(x_test)
# print('Classes', classes)
# print('Probas', probas)

class Form:
    def __init__(self, arg):
        """[summary]

        Arguments:
            arg {path or np.ndarray} -- if passing np.ndarray, image must be gray mode (one channel)
        """

        if type(arg) is str:
            self.image = cv2.imread(arg, 0)
            self.original = cv2.imread(arg)
        elif type(arg) is np.ndarray:
            self.image = arg

    def __del__(self):
        cv2.destroyAllWindows()

    def preprocess(self):
        self.prep_obj = Preprocess(
            self.image, square_size=57, vertical_crop=(600, 3000))
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
        return [row[:, 260:930], row[:, 980:1072], row[:, 1094:1182], row[:, 1200:1290], row[:, 1350:1436], row[:, 1460:1542]]

    @staticmethod
    def bin_reverse(img):
        zeros = img == 0
        nonzeros = img != 0
        img[zeros] = 1
        img[nonzeros] = 0
        return img

    @staticmethod
    def is_blank(img):
        zeros = img == 0
        nonzeros = img != 0
        img[zeros] = 1
        img[nonzeros] = 0
        return img

    @staticmethod
    def extract_number(img):
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # # print(hsv.shape)
        # # print(hsv)
        # # print(hsv[:, :, 2]<20)
        # Let's define our kernel size
        kernel = np.ones((5, 5), np.uint8)

        # moving dilate filter
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


        # hsv[hsv[:, :, 2]<20] = 0
        # # hsv[hsv[:, :, 1]<20] = 0
        # # hsv[hsv[:, :, 0]<20] = 0


        # print(hsv.shape)
        # print(hsv)
        # cv2.imshow('kj', img)
        # cv2.waitKey()
        return img
        
    
    @staticmethod
    def gray(img):
        pass

    def get_rows(self, margin_top=55, margin_bottom=45):
        for row in self.prep_obj.rows:
            row_img = self.image[
                int(self.prep_obj.kwargs['vertical_crop'][0] + row - margin_top):
                int(self.prep_obj.kwargs['vertical_crop'][0] + row + margin_bottom)]

            text, *digits = self.segment_row(row_img)

            digits = list(map(self.extract_number, digits))

            digits = list(map(lambda x: cv2.resize(x, (20, 20)), digits))
            # print(len(digits))
            # print(digits[0].shape)
            digits = list(map(lambda x: np.reshape(x, (20, 20, 1)), digits))

            # # Preparing for feeding to cnn
            digits = list(map(lambda x: x/255, digits))
            # # Reverse image
            digits = [self.bin_reverse(digit) for digit in digits]

            show('row', row_img)
            for d in digits:
                # show('extracted', self.extract_number(d))
                show('row', d)
                # print(digits)
                # print(len(digits))
                # print(digits[0].shape)
                classes_, probas_ = model.predict_single(d)
                print(classes_)


if __name__ == '__main__':
    Form('/home/mohammad/Desktop/Forms/form8/00000015.jpg').preprocess().get_rows()
