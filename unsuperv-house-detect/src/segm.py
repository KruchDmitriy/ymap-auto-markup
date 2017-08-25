import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from random import randint


img = cv2.imread("../resources/map2.png")
# plt.imshow(img)
# plt.show()

print(img.shape)
print(img.dtype)
X = np.zeros(shape=(img.shape[0] * img.shape[1], 5), dtype=img.dtype)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        X[i * img.shape[1] + j][0] = img[i][j][0]
        X[i * img.shape[1] + j][1] = img[i][j][1]
        X[i * img.shape[1] + j][2] = img[i][j][2]
        X[i * img.shape[1] + j][3] = 2 * i
        X[i * img.shape[1] + j][4] = 2 * j

params = {"eps": 4, "min_samples": 12}
threshold = 50 # pixels

clust = DBSCAN(*params.values()).fit(X)

print("params = " + str(params))
n_clusters_ = len(set(clust.labels_))# - (1 if -1 in clust.labels_ else 0)
print("num_clusters = " + str(n_clusters_))


good_clusters = []
max_cluster = -1
idx_max_cluster = 0
for cluster in range(n_clusters_):
    num_entities = np.count_nonzero(clust.labels_ == cluster)
    if num_entities > threshold:
        good_clusters.append(cluster)

        if num_entities > max_cluster:
            max_cluster = num_entities
            idx_max_cluster = cluster


# good_clusters.remove(idx_max_cluster)
good_clusters = set(good_clusters)
print("num_clusters after filtration = " + str(len(good_clusters)))

colors = []
for cluster in range(n_clusters_):
    colors.append([randint(0, 255), randint(0, 255), randint(0, 255)])

colors[-1] = [0, 0, 0]


dst = np.zeros(shape=img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        label = clust.labels_[i * img.shape[1] + j]
        if label in good_clusters:
            dst[i][j] = colors[label]
        else:
            dst[i][j] = colors[-1]

plt.imshow(dst)
plt.show()
