import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import listdir


PATH_TO_IMG = "../preprocessedData/imgs/"
PATH_TO_LABELS = "../preprocessedData/labels/"
PATH_TO_DIST_LABELS = "../preprocessedData/dist_labels/"


def signed_dist_trans(img):
    dst, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(shape=img.shape)

    ret = np.zeros(shape=dst.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist = -np.inf
            for contour in contours:
                cur_dist = cv2.pointPolygonTest(contour, (j, i), True)
                if cur_dist > dist:
                    dist = cur_dist

            if dist < 0:
                continue

            ret[i][j] = dist

    # f = plt.figure(1)
    # plt.imshow(ret)
    # g = plt.figure(2)
    # plt.imshow(img)
    # plt.show()

    dst = cv2.normalize(ret, dst, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return dst

def main():
    with open('features_test.txt', 'w') as feat:
        for file in listdir(PATH_TO_IMG):
            label = cv2.imread(PATH_TO_LABELS + file, 0)

            result = signed_dist_trans(label)
            cv2.imwrite(PATH_TO_DIST_LABELS + file, result)
            print(file)


if __name__ == "__main__":
    main()
