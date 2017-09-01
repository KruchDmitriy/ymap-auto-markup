import numpy as np
import cv2
from numpy.linalg import norm


def grad_descent(grad, lrate=0.001):
    print('grad desc')
    x = np.abs(np.random.normal(0, 1, grad.shape[1]))
    x[x > 1.] == 1.
    eps = 1.

    n_iter = 0
    while eps > 1e-5:
        x_prev = x
        x = x + np.dot(grad, x) * lrate
        x[x < 0.] == 0.
        x[x > 1.] == 1.
        eps = norm(x - x_prev)

        n_iter += 1
        if n_iter % 100 == 0:
            print(str(n_iter) + " " + str(eps))

    print('grad desc finished')
    return x_prev


def calc_mean(probas, s, bbox_height, bbox_width):
    bbox_half_height = bbox_height // 2
    bbox_half_width = bbox_width // 2

    height = probas.shape[0] - 1
    width = probas.shape[1] - 1

    dst_shape = ((height + 1) * bbox_half_height, (width + 1) * bbox_half_width)
    dst = np.zeros(shape=dst_shape)

    for x in range(height):
        for y in range(width):
            i = x * bbox_half_height
            j = y * bbox_half_width

            dst[i: i + bbox_height, j: j + bbox_width] += probas[x][y] * s[x * width + y]

    for x in range(height + 1):
        for y in range(width + 1):
            if x == 0 and y == 0 or x == 0 and y == width or\
             x == height and y == 0 or x == height and y == width:
                neighbs = 1.
            elif x == 0 or y == 0 or x == height or y == width:
                neighbs = 2.
            else:
                neighbs = 4.

            i = x * bbox_half_height
            j = y * bbox_half_width
            dst[i: i + bbox_half_height, j: j + bbox_half_width] /= neighbs

    return dst


def count_neigbs(x, y, width, height):
    return int(x > 0) + int(x < (height - 1)) + int(y > 0) + int(y < (width - 1))


def norm_grad(probas, bbox_height, bbox_width):
    bbox_half_height = bbox_height // 2
    bbox_half_width = bbox_width // 2

    height = probas.shape[0] - 1
    width = probas.shape[1] - 1

    L = width * height
    N = np.eye(L) * -1.

    for x in range(height):
        for y in range(width):
            neighbs = count_neigbs(x, y, width, height)
            norm = np.sum(probas[x][y] * probas[x][y]) * neighbs

            if x > 0:
                neighb = probas[x - 1][y][bbox_half_height:]
                cur = probas[x][y][:bbox_half_height]

                N[x * width + y][(x - 1) * width + y] = np.sum(neighb * cur) / norm
            if x < height - 1:
                neighb = probas[x + 1][y][:bbox_half_height]
                cur = probas[x][y][bbox_half_height:]

                N[x * width + y][(x + 1) * width + y] = np.sum(neighb * cur) / norm

            if y > 0:
                neighb = probas[x][y - 1][:, bbox_half_width:]
                cur = probas[x][y][:, :bbox_half_width]

                N[x * width + y][x * width + y - 1] = np.sum(neighb * cur) / norm

            if y < width - 1:
                neighb = probas[x][y + 1][:, :bbox_half_width]
                cur = probas[x][y][:, bbox_half_width:]

                N[x * width + y][x * width + y + 1] = np.sum(neighb * cur) / norm

    s = grad_descent(N)
    dst = calc_mean(probas, s, bbox_height, bbox_width)
    dst = cv2.normalize(dst, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return dst


def norm_max(probas, bbox_height, bbox_width, img_height, img_width):
    bbox_half_height = bbox_height // 2
    bbox_half_width = bbox_width // 2

    dst = np.zeros(shape=(img_height, img_width, 3), dtype='uint8')

    for i in np.arange(0, img_height - bbox_height, bbox_half_height):
        for j in np.arange(0, img_width - bbox_width, bbox_half_width):
            for x in range(bbox_width):
                for y in range(bbox_height):
                    val = max(dst[i + x][j + y][0], int(255 * probas[i, j, x, y]))
                    dst[i + x][j + y] = np.array([val, val, val])

    return dst
