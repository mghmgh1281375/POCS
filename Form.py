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
model.load_weights('models/model_conv_20181024081423.h5')
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
        return [row[:, 260:930], row[:, 970:1072], row[:, 1080:1182], row[:, 1200:1290], row[:, 1340:1440], row[:, 1450:1546]]

    def get_rows(self, margin=65):
        for row in self.prep_obj.rows:
            row_img = self.image[
                int(self.prep_obj.kwargs['vertical_crop'][0] + row - margin):
                int(self.prep_obj.kwargs['vertical_crop'][0] + row + margin)]

            text, *digits = self.segment_row(row_img)
            digits = list(map(lambda x: cv2.resize(x, (20, 20)), digits))
            digits = list(map(lambda x: np.reshape(x, (20, 20, 1)), digits))
            digits = list(map(lambda x: x/255, digits))

            # print(digits[2])
            show('row', digits[2])
            


            # print(digits)
            # print(len(digits)) 
            # print(digits[0].shape)
            classes_, probas_ = model.predict_single(digits[2])
            print(classes_)
            show('row', row_img)
            # show('segment', )


if __name__ == '__main__':
    Form('/home/mohammad/Desktop/Forms/form8/00000015.jpg').preprocess().get_rows()
