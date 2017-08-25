import cv2
import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


# img = cv2.imread('../house-detect/resources/test2.jpg')
img = np.zeros(shape=(64, 64, 3), dtype="uint8")
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if i <= j:
            img[i][j] = [255, 255, 255]

plt.imshow(img)
plt.show()

img_norm = np.copy(img)
img_norm = cv2.normalize(img, dst=img_norm, alpha=0., beta=1.,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

n_pix = img.shape[0] * img.shape[1]
height = img.shape[0]
width = img.shape[1]

vector_img = img_norm.reshape((n_pix, img.shape[2]))
dist_mat = cdist(vector_img, vector_img, 'euclidean')

weights = np.zeros(shape=(n_pix, n_pix))
diag = np.zeros(shape=(n_pix, n_pix))

for i in range(n_pix):
    sum1 = 0.
    sum2 = 0.
    for x in [-1, 0, 1]:
        for y in [-width, 0, width]:
            if x == 0 and y == 0 or i + x + y < 0 or i + x + y >= n_pix:
                continue

            sum1 += dist_mat[i][i + x + y]
            sum2 += dist_mat[i][i + x + y] * dist_mat[i][i + x + y]

    # disp = sum2 - sum1 * sum1
    disp = 1.
    # if disp < 1e-20: disp = 1e-20

    # for x in [-3, -2, -1, 0, 1, 2, 3]:
    #     for y in [-3 * width, -2 * width, -width, 0, width, 2 * width, 3 * width]:
    for x in [-1, 0, 1]:
        for y in [-width, 0, width]:
            if x == 0 and y == 0 or i + x + y < 0 or i + x + y >= n_pix:
                continue

            y_ = y / width
            dist = np.exp(- (x * x + y_ * y_) / 25)

            weights[i][i + x + y] = np.exp(-dist_mat[i][i + x + y] * dist_mat[i][i + x + y] / disp) * dist
            # if weights[i][i + x + y]

    diag[i][i] = sum1

# weights[np.isnan(weights)] = 0.
lap = diag - weights
print(weights)
print(lap)
plt.imshow(lap, cmap='hot', interpolation='nearest')
plt.show()

# for i in range(n_pix):
#     lap[i][i] = 0.
#     if i != 0:
#         lap[i][i - 1] = -dist_mat[i][i - 1]
#         lap[i][i] += dist_mat[i][i - 1]
#     if i != n_pix - 1:
#         lap[i][i + 1] = -dist_mat[i][i + 1]
#         lap[i][i] += dist_mat[i][i + 1]

#     if i >= width:
#         lap[i][i - width] = -dist_mat[i][i - width]
#         lap[i][i] += dist_mat[i][i - width]

#     if i < width * (width - 1):
#         lap[i][i + width] = -dist_mat[i][i + width]
#         lap[i][i] += dist_mat[i][i + width]

    # if i != 0:
    #     lap[i][i - 1] /= lap[i][i]
    # if i != n_pix - 1:
    #     lap[i][i + 1] /= lap[i][i]

    # if i >= width:
    #     lap[i][i - width] /= lap[i][i]

    # if i < width * (width - 1):
    #     lap[i][i + width] /= lap[i][i]

    # lap[i][i] = 1.

# lap = np.eye(n_pix) * 4. - lap
print("similarity matrix was built")

lambdas, vectors = eig(lap)
# print(lambdas, vectors)

# lam_idx = np.argpartition(lambdas, 2)[1]
# vec = vectors[lam_idx]# / np.linalg.norm(vectors[lam_idx])
# norm = np.linalg.norm(vectors[lam_idx])

# plt.plot(vec)
# plt.show()


dst = np.zeros(shape=(img.shape[0] * img.shape[1]))

for k in range(len(vectors)):
    # if lambdas[k] > 0:
    dst += 1. / np.abs(lambdas[k]) * vectors[k]

# for i in range(len(vectors)):
#     if lambdas[i] > 0:
#         for j in range(n_pix):
#             x = int(j / width)
#             y = int(j % width)

#             dst[x][y] += 1. / np.sqrt(lambdas[i] + 0.0001) * vectors[i][j]

dst.reshape((img.shape[0], img.shape[1]))
dst_norm = np.zeros(shape=dst.shape)
dst_norm = cv2.normalize(dst, dst=dst_norm, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

plt.axis("off")
# plt.imshow(cv2.cvtColor(dst_norm, cv2.COLOR_BGR2RGB))
plt.imshow(dst_norm)
plt.show()
