import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from random import randint
from numpy.random import permutation


PATH_TO_IMG = "../preprocessedData/imgs/"
PATH_TO_LABELS = "../preprocessedData/labels/"

BBOX_WIDTH = 64
BBOX_HEIGHT = 64

BBOX_HALF_WIDTH = BBOX_WIDTH // 2
BBOX_HALF_HEIGHT = BBOX_HEIGHT // 2

TEST_IMAGE = "../preprocessedData/imgs/3band_AOI_1_RIO_img4599.png"
# MODE = "TEST"
MODE = "CREATE"

def main():
    hog_params = {
        "win_size": (BBOX_HEIGHT, BBOX_WIDTH),
        "block_size": (16, 16),
        "block_stride": (8, 8),
        "cell_size": (8, 8),
        "nbins": 9,
        "deriv_aperture": 3,
        "win_sigma": 4.,
        "histogram_norm_type": 0,
        "L2_hys_threshold": 0.2,
        "gamma_correction": True,
        "nlevels": 64
    }

    hog = cv2.HOGDescriptor(*hog_params.values())
    print(hog.getDescriptorSize())
    input()
    # return

    win_stride = (8, 8)
    padding = (0, 0)

    # plt.axis("off")
    # plt.imshow(image, interpolation='none')
    # plt.show()

    idx = 0

    if MODE == 'TEST':
        with open('features_hog_house.txt', 'w') as hog_house:
            img = cv2.imread(TEST_IMAGE)
            for i in np.arange(0, img.shape[0] - BBOX_HEIGHT, BBOX_HALF_HEIGHT):
                for j in np.arange(0, img.shape[1] - BBOX_WIDTH, BBOX_HALF_WIDTH):
                    print(i, j)
                    hist = hog.compute(img, win_stride, padding, locations=((int(i), int(j)),))
                    hist = hist.reshape((hist.shape[0],))
                    for x in range(BBOX_WIDTH):
                        for y in range(BBOX_HEIGHT):
                            vec = np.zeros(shape=(hog.getDescriptorSize() + 5), dtype='float64')
                            vec[:-5] = hist
                            vec[-5] = x
                            vec[-4] = y
                            vec[-3] = img[i + x][j + y][0]
                            vec[-2] = img[i + x][j + y][1]
                            vec[-1] = img[i + x][j + y][2]

                            vec.tofile(hog_house, sep='\t')
                            hog_house.write('\n')
    else:
        with open('features_hog_full.txt', 'w') as feat:
            n_samples = 0
            for file in permutation(listdir(PATH_TO_IMG)):
                img = cv2.imread(PATH_TO_IMG + file)
                label = cv2.imread(PATH_TO_LABELS + file)

                for count in range(30):
                    i = randint(0, img.shape[0] - 1 - BBOX_HEIGHT)
                    j = randint(0, img.shape[1] - 1 - BBOX_WIDTH)

                    hist = hog.compute(img, win_stride, padding, locations=((i, j),))
                    hist = hist.reshape((hist.shape[0],))

                    for sample in range(50):
                        n_samples += 1
                        vec = np.zeros(shape=(hog.getDescriptorSize() + 5))

                        x = randint(0, BBOX_HEIGHT)
                        y = randint(0, BBOX_WIDTH)

                        vec[:-5] = hist
                        vec[-5] = x
                        vec[-4] = y
                        vec[-3] = img[i + x][j + y][0]
                        vec[-2] = img[i + x][j + y][1]
                        vec[-1] = img[i + x][j + y][2]

                        lab = int((label[i + x][j + y] == [255, 255, 255])[0])
                        str_id = str(idx)

                        feat.write(str_id + '\t' + str(lab) + '\t' + str_id + '\t0\t')
                        vec.tofile(feat, sep='\t')
                        feat.write('\n')

                        idx += 1
            if n_samples % 100 == 0:
                print(n_samples)


if __name__ == "__main__":
    main()
