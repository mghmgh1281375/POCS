"""  """

import numpy as np
import cv2

class Filter:
    def __init__(self, size):
        pass

class Image:
    def __init__(self, path):
        self.image = cv2.imread(path, 0)
        cv2.imshow('original', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(self.image.shape)

    def resize(self, size):
        self.old_height, self.old_width = self.image.shape
        self.new_height, self.new_width = size
        kernel = np.ones(((self.old_height - self.new_height) + 1, (self.old_width - self.new_width) + 1))
        return self.convolve(self.image, kernel)

    def convolve(self, image, kernel):
        new_image = [[0] * self.new_width] * self.new_height
        for i in range(image.shape[0] - kernel.shape[0] + 1):
            for j in range(image.shape[1] - kernel.shape[1] + 1):
                p = image[i: i + kernel.shape[0], j: j + kernel.shape[1]]
                new_image[i][j] = np.sum(kernel*p)/(kernel.shape[0]*kernel.shape[1])
        return np.array(new_image, dtype='uint8')

if __name__ == "__main__":
    new_image = Image('download.jpeg').resize((275, 183))
    cv2.imshow('resized (150, 85)', new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    