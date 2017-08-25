import cv2
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt


house = cv2.imread('../resources/house2.png')
house_gray = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()
kp, des = fast.detectAndCompute(house_gray, None)

print(kp)

fast_house = np.zeros(house_gray.shape)
fast_house = cv2.drawKeypoints(image=house_gray, keypoints=kp, outImage=fast_house)
cv2.imwrite('fast_keypoints.jpg', fast_house)


img = cv2.imread('../resources/map2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp_img, des_img = fast.detectAndCompute(img_gray, None)

# print(kp_img)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des, des_img, k=2)

good = []
for m,n in matches:
    # if m.distance < 0.75 * n.distance:
    good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = np.zeros(shape=(img.shape[0] + house.shape[0], img.shape[1] + house.shape[1]))
img3 = cv2.drawMatchesKnn(house_gray, kp, img_gray, kp_img, good, outImg=img3, flags=2)
plt.imshow(img3)
plt.show()

# matching_img = np.vstack(
#     (
#         np.hstack((house, np.zeros(shape=(house.shape[0], img.shape[1], 3)))),
#         np.hstack((np.zeros(shape=(img.shape[0], house.shape[1], 3)), img))
#     )
# )

# for i in range(len(src_points)):
#     pt1 = src_points[i]
#     pt2 = dst_points[i]
#     from_ = (int(pt2[0]), int(pt2[1]))
#     to_ = int(house.shape[0] + pt1[0]), int(house.shape[1] + pt1[1])
#     print(from_, to_)

#     pt = (to_[0] - from_[0], to_[1] - from_[1])
#     # cv2.line(matching_img, from_, to_, (0, 255, 255))
#     cv2.rectangle(matching_img, pt,
#         (pt[0] + house.shape[0], pt[1] + house.shape[1]), (0, 0, 255))

# cv2.imwrite('matching.jpg', matching_img)


# clust = DBSCAN(eps=30, min_samples=4).fit(src_points)
# # clust = MeanShift().fit(src_points)

# n_clusters_ = len(set(clust.labels_)) - (1 if -1 in clust.labels_ else 0)

# clusters = []
# print("num clusters = ", n_clusters_)

# dst_points = np.array(dst_points)
# src_points = np.array(src_points)

# for label in range(n_clusters_):
#     dst = dst_points[clust.labels_ == label]
#     src = src_points[clust.labels_ == label]

#     if len(dst) > THRESHOLD_NUM_POINTS:
#         res = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
#         print(label)
#         print(res)
#         M = res[0]
#         mask = res[1]

#         if M == None:
#             continue
#         # print(M)
#         # matchesMask = mask.ravel().tolist()

#         h,w = house_gray.shape
#         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#         src = cv2.perspectiveTransform(pts,M)

#         img = cv2.polylines(img,[np.int32(src)],True,255,3, cv2.LINE_AA)

# cv2.imwrite('sift_result.jpg', img)
