import numpy as np
import cv2

class Preprocess:
    def __init__(self, image, **kwargs):
        self.image = image
        self.kwargs = kwargs
        print(kwargs)

        # cropping
        self.image = self.image[self.kwargs['vertical_crop'][0]:self.kwargs['vertical_crop'][1], :]

        # Binarizing image
        self.image[self.image < 245] = 0
        self.image[self.image >= 245] = 255

        # Let's define our kernel size
        kernel = np.ones((5,5), np.uint8)

        # moving dilate filter
        self.image = cv2.dilate(self.image, kernel, iterations = 3)

    def find_rows(self):
        """find rows
        
        Returns:
            [list] -- return rows centers
        """

        project = np.sum(self.image, axis=1)
        black_pixels = np.argwhere(project < project[int(self.image.shape[1]/2)])
        black_pixels = np.reshape(black_pixels, (black_pixels.shape[0],))

        black_pixel_boundries = []
        black_pixel_boundries.append(black_pixels[0])
        tmp = black_pixels[0]
        for i in range(1, len(black_pixels)):
            if black_pixels[i] - 1 == tmp:
                tmp = black_pixels[i]
            else:
                black_pixel_boundries.append(tmp)
                black_pixel_boundries.append(black_pixels[i])
                tmp = black_pixels[i]
        black_pixel_boundries.append(tmp)

        rows_boundries = np.reshape(black_pixel_boundries, (int(len(black_pixel_boundries)/2), 2))
        self.rows = np.absolute(np.mean(rows_boundries, axis=1, dtype=np.uint64))
        return self

    def find_cols_boundries(self):
        project = np.sum(self.image, axis=0)
        black_pixels = np.argwhere(project < project[int(self.image.shape[0]/2)])
        black_pixels = np.reshape(black_pixels, (black_pixels.shape[0],))
        black_pixel_boundries = []
        black_pixel_boundries.append(black_pixels[0])
        tmp = black_pixels[0]
        for i in range(1, len(black_pixels)):
            if black_pixels[i] - 1 == tmp:
                tmp = black_pixels[i]
            else:
                black_pixel_boundries.append(tmp)
                black_pixel_boundries.append(black_pixels[i])
                tmp = black_pixels[i]
        black_pixel_boundries.append(tmp)

        print('column boundries', black_pixel_boundries)
        self.cols_boundries = np.reshape(black_pixel_boundries, (2, 2))
        return self
