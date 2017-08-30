from cpp_ottree.tree import Ensemble
import numpy as np
from numpy.linalg import norm
import cv2
import time
from matplotlib import pyplot as plt
from os.path import exists


PATH_TO_IMG = "../preprocessedData/imgs/"
PATH_TO_LABELS = "../preprocessedData/labels/"

BBOX_WIDTH = 64
BBOX_HEIGHT = 64

BBOX_HALF_WIDTH = BBOX_WIDTH // 2
BBOX_HALF_HEIGHT = BBOX_HEIGHT // 2

TEST_IMAGE = PATH_TO_IMG + "3band_AOI_1_RIO_img4599.png"
RESULT_IMAGE = "./hog_10K.png"
LABEL_IMAGE = PATH_TO_LABELS + "3band_AOI_1_RIO_img4599.png"


def iou(src, lab):
    src = (src == 255)
    lab = (lab == 255)

    intersection = np.count_nonzero(np.logical_and(src, lab))
    union = np.count_nonzero(np.logical_or(src, lab))

    print(intersection)
    print(union)

    return intersection / union


def grad_descent(grad, lrate=0.001):
    print('grad desc')
    x = np.abs(np.random.normal(0, 1, grad.shape[1]))
    x[x > 1.] == 1.
    eps = 1.

    n_iter = 0
    while eps > 1e-5:
        x_prev = x
        # x = x + np.dot(grad, np.exp(x)) * np.exp(x) * lrate
        x = x + np.dot(grad, x) * lrate
        x[x < 0.] == 0.
        x[x > 1.] == 1.
        eps = norm(x - x_prev)

        n_iter += 1
        # print(eps)
        if n_iter % 100 == 0:
            print(str(n_iter) + " " + str(eps))

    print('grad desc finished')
    return x_prev


def calc_mean(probas, s):
    height = probas.shape[0] - 1
    width = probas.shape[1] - 1

    dst_shape = ((height + 1) * BBOX_HALF_HEIGHT, (width + 1) * BBOX_HALF_WIDTH)
    dst = np.zeros(shape=dst_shape)

    for x in range(height):
        for y in range(width):
            i = x * BBOX_HALF_HEIGHT
            j = y * BBOX_HALF_WIDTH

            dst[i: i + BBOX_HEIGHT, j: j + BBOX_WIDTH] += probas[x][y] * s[x * width + y]

    for x in range(height + 1):
        for y in range(width + 1):
            neighbs = 0.

            if x == 0 and y == 0 or x == 0 and y == width or\
             x == height and y == 0 or x == height and y == width:
                neighbs = 1.
            elif x == 0 or y == 0 or x == height or y == width:
                neighbs = 2.
            else:
                neighbs = 4.

            i = x * BBOX_HALF_HEIGHT
            j = y * BBOX_HALF_WIDTH
            dst[i: i + BBOX_HALF_HEIGHT, j: j + BBOX_HALF_WIDTH] /= neighbs

    # print(dst)

    return dst


def count_neigbs(x, y, width, height):
    return int(x > 0) + int(x < (height - 1)) + int(y > 0) + int(y < (width - 1))


def normalize(probas):
    height = probas.shape[0] - 1
    width = probas.shape[1] - 1

    # ATTENTION! TEST!
    # arr = np.random.normal(0, 1, probas.shape[0] * probas.shape[1])
    # for i in range(probas.shape[0]):
    #     for j in range(probas.shape[1]):
    #         probas[i][j] = np.ones(shape=(probas.shape[2], probas.shape[3])) * arr[i * probas.shape[1] + j]

    # s = np.ones(height * width)
    # test = calc_mean(probas, s)
    # plt.axis('off')
    # plt.imshow(test)
    # plt.show()

    r_width = probas.shape[1]

    L = width * height
    N = np.eye(L) * -1.

    for x in range(height):
        for y in range(width):
            neighbs = count_neigbs(x, y, width, height)
            norm = np.sum(probas[x][y] * probas[x][y]) * neighbs

            if x > 0:
                neighb = probas[x - 1][y][BBOX_HALF_HEIGHT:]
                cur = probas[x][y][:BBOX_HALF_HEIGHT]

                N[x * width + y][(x - 1) * width + y] = np.sum(neighb * cur) / norm
            if x < height - 1:
                neighb = probas[x + 1][y][:BBOX_HALF_HEIGHT]
                cur = probas[x][y][BBOX_HALF_HEIGHT:]

                N[x * width + y][(x + 1) * width + y] = np.sum(neighb * cur) / norm

            if y > 0:
                neighb = probas[x][y - 1][:, BBOX_HALF_WIDTH:]
                cur = probas[x][y][:, :BBOX_HALF_WIDTH]

                N[x * width + y][x * width + y - 1] = np.sum(neighb * cur) / norm

            if y < width - 1:
                neighb = probas[x][y + 1][:, :BBOX_HALF_WIDTH]
                cur = probas[x][y][:, BBOX_HALF_WIDTH:]

                N[x * width + y][x * width + y + 1] = np.sum(neighb * cur) / norm

    # plt.imshow(N)
    # plt.show()
    s = grad_descent(N)
    input()

    dst = calc_mean(probas, s)

    return dst

def main():
    # if exists(RESULT_IMAGE):
    #     img = cv2.imread(RESULT_IMAGE)

    #     one_channel = img[:, :, 0]
    #     thresh, dst = cv2.threshold(one_channel, 70, 255, cv2.THRESH_BINARY)
    #     print(thresh)

    #     label = cv2.imread(LABEL_IMAGE)
    #     lab_one_channel = label[:, :, 0]

    #     dst_ = np.zeros(lab_one_channel.shape)
    #     dst_[:dst.shape[0], :dst.shape[1]] = dst

    #     print("IoU = ", iou(dst_, lab_one_channel))

    #     plt.axis('off')
    #     plt.imshow(dst)
    #     plt.show()
    #     return 0

    ensemble = Ensemble('features_hog_circles_test.txt.model', 6, False)

    hog_params = {
        "win_size": (BBOX_HEIGHT, BBOX_WIDTH),
        "block_size": (16, 16),
        "block_stride": (16, 16),
        "cell_size": (8, 8),
        "nbins": 9,
        "deriv_aperture": 1,
        "win_sigma": 4.,
        "histogram_norm_type": 0,
        "L2_hys_threshold": 0.2,
        "gamma_correction": 0,
        "nlevels": 64
    }
    win_stride = (8, 8)
    padding = (0, 0)

    hog = cv2.HOGDescriptor(*hog_params.values())

    img = cv2.imread(TEST_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    # plt.imshow(img)
    # plt.show()


    dst = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype='uint8')
    probas = np.zeros(shape=(img.shape[0] // BBOX_HALF_HEIGHT,
                             img.shape[1] // BBOX_HALF_WIDTH,
                             BBOX_HEIGHT, BBOX_WIDTH))

    # dump_probas_file = 'probas.pkl'
    # if exists(dump_probas_file):
    #     probas = np.load(dump_probas_file)
    # else:
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

                    # start = time.time()
                    value = 1. / (1. + np.exp(-ensemble.predict(vec)))
                    probas[i // BBOX_HALF_HEIGHT][j // BBOX_HALF_WIDTH][x][y] = value

                    # end = time.time()
                    # print('elapsed time: ', end - start)

                    # val = max(dst[i + x][j + y][0], int(255 * value))
                    # dst[i + x][j + y] = np.array([val, val, val])
        # probas.dump(dump_probas_file)

    dst = normalize(probas)

    dst = cv2.normalize(dst, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    cv2.imwrite('hog_test.png', dst)
    plt.imshow(dst)
    plt.show()


if __name__ == "__main__":
    main()
