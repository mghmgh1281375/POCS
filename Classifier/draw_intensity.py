
import numpy as np
import cv2
import matplotlib.pyplot as plt

X_info = {
    'width': 50,
    'height': 100,
    'color': (20, 20, 20)
}
Y_info = {
    'width': 100,
    'height': 50,
    'color': (20, 20, 20)
}
sample_path = 'Classifier/2-78.png'
sample = cv2.imread(sample_path, 0)
print(sample.shape)
sample = cv2.resize(sample, (100, 100))
ret,sample = cv2.threshold(sample, 127, 255, cv2.THRESH_BINARY_INV)


def normalize(x):
    return x/max(x)

X = np.sum(sample, axis=0)
Y = np.sum(sample, axis=1)

X_normalized = normalize(X)
Y_normalized = normalize(Y)


X_label = np.ones((50, 100))*255
Y_label = np.ones((100, 50))*255

tmp = X_normalized*50
for j in range(100):
    for i in range(50):
        if i < tmp[j]:
            X_label[i][j] = 0

tmp = Y_normalized*50
for i in range(100):
    for j in range(50):
        if j < tmp[i]:
            Y_label[i][j] = 0


XY_normalized = np.concatenate((X_normalized, Y_normalized))

X_label = np.ones((50, 100))*255
Y_label = np.ones((100, 50))*255
XY = np.ones((50, 200))*255

tmp = X_normalized*50
for j in range(100):
    for i in range(50):
        if i < tmp[j]:
            X_label[i][j] = 0

tmp = Y_normalized*50
for i in range(100):
    for j in range(50):
        if j < tmp[i]:
            Y_label[i][j] = 0

tmp = XY_normalized*50
for i in range(200):
    for j in range(50):
        if j < tmp[i]:
            XY[j][i] = 0

cv2.imwrite('Classifier/X.png', X_label)
cv2.imwrite('Classifier/Y.png', Y_label)
cv2.imwrite('Classifier/XY.png', XY)
cv2.imwrite('Classifier/sample.png', sample)

# cv2.waitKey()
# cv2.destroyAllWindows()




