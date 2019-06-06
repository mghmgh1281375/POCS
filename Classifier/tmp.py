import cv2
import numpy as np

def focus(img):
    X = np.sum(img, axis=0)
    Y = np.sum(img, axis=1)
    x_min, x_max, y_min, y_max = None, None, None, None

    for i in range(X.shape[0]):
        if X[i] != 0 and x_min is None:
            x_min = i
        if X[-i] != 0 and x_max is None:
            x_max = -i
    for i in range(Y.shape[0]):
        if Y[i] != 0 and y_min is None:
            y_min = i
        if Y[-i] != 0 and y_max is None:
            y_max = -i
    print(y_min, y_max, x_min, x_max)
    return img[y_min:y_max, x_min:x_max]

org = cv2.imread('Classifier/2-78.png', 0)
dilated = cv2.dilate(org, np.ones((3, 3), np.uint8), iterations=1)
_, img = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)
img = focus(img)
img = cv2.resize(img, (50, 50))

cv2.imshow('org', org)
cv2.imshow('dilated', dilated)
cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyAllWindows()

# features = feature_extractor(img)
