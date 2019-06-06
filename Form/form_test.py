import cv2
import numpy as np

from time import time
def timeit(func):
    def call(*args, **kwargs):
        t0 = time()
        ret = func(*args, **kwargs)
        t1 = time()
        print(t1 - t0)
        return ret
    return call

SQUARE_SIZE = 57

cv2.namedWindow('form', cv2.WINDOW_GUI_NORMAL)
def show(img):
    cv2.imshow('form', img)
    cv2.waitKey()

form = cv2.imread('/home/mohammad/Desktop/Forms/form1/00000010.jpg', 0)
print('image shape', form.shape)

def rows_centers(img):
    project = np.sum(img, axis=1)
    poses = np.argwhere(project < project[int(img.shape[1]/2)])
    poses = np.reshape(poses, (poses.shape[0],))
    # print(poses)
    cols_position = []
    cols_position.append(poses[0])
    tmp = poses[0]
    for i in range(1, len(poses)):
        if poses[i] - 1 == tmp:
            tmp = poses[i]
        else:
            cols_position.append(tmp)
            cols_position.append(poses[i])
            tmp = poses[i]
    cols_position.append(tmp)

    # print('rows position', cols_position)
    rows_boundries = np.reshape(cols_position, (int(len(cols_position)/2), 2))
    
    return np.absolute(np.mean(rows_boundries, axis=1, dtype=np.uint64))



def square_positions(img):
    show(img)
    project = np.sum(img, axis=0)
    poses = np.argwhere(project < project[int(img.shape[0]/2)])
    poses = np.reshape(poses, (poses.shape[0],))
    cols_position = []
    cols_position.append(poses[0])
    tmp = poses[0]
    for i in range(1, len(poses)):
        if poses[i] - 1 == tmp:
            tmp = poses[i]
        else:
            cols_position.append(tmp)
            cols_position.append(poses[i])
            tmp = poses[i]
    cols_position.append(tmp)

    print('columns position', cols_position)
    return cols_position

def measeure_angle(img):
    
    a= square_positions(img[0                       :int(img.shape[0]/6), :])
    # b= square_positions(img[int(img.shape[0]/4)     :int(img.shape[0]/2), :])
    # c= square_positions(img[int(img.shape[0]/2)     :3*int(img.shape[0]/4), :])
    d= square_positions(img[5*int(img.shape[0]/6)   :int(img.shape[0]), :])

    rotate_samt = 0
    if a[1] > d[1]:
        rotate_samt = 1
    else:
        rotate_samt = -1
    
    print(a[0] - d[0], a[1] - d[1], a[2] - d[2], a[3] - d[3])


@timeit
def preprocessing(img):

    # cropping
    img = img[600:3000, :]

    # Binarizing image
    img[img < 245] = 0
    img[img >= 245] = 255

    # Let's define our kernel size
    kernel = np.ones((5,5), np.uint8)

    # moving dilate filter
    img = cv2.dilate(img, kernel, iterations = 3)

    # poses = square_positions(img)
    rows_centers(img)

    
    # edges = cv2.Canny(img,50,150,apertureSize = 3)

    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLines(edges,1,np.pi/180,200, minLineLength, maxLineGap)
    # if lines is not None:
    #     for x1,y1,x2,y2 in lines[0]:
    #         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    return img





show(preprocessing(form))





cv2.destroyAllWindows()