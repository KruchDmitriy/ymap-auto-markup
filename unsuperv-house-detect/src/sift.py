import cv2
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN


THRESHOLD = 300.
THRESHOLD_NUM_POINTS = 2

def nearest(vec, keypoints, descriptors):
    neigbor = -1
    min_dist = 1e6

    for i in range(descriptors.shape[0]):
        pt = keypoints[i]
        v = descriptors[i]
        d = np.linalg.norm(vec - v)

        if d < min_dist:
            min_dist = d
            neigbor = i

    if min_dist < THRESHOLD:
        return neigbor

    return -1


def find_pairs(keypoints1, descriptors1, keypoints2, descriptors2):
    src_points = []
    dst_points = []

    for i in range(descriptors1.shape[0]):
        pt1 = keypoints1[i]
        desc1 = descriptors1[i]
        nn = nearest(desc1, keypoints2, descriptors2)

        if nn >= 0:
            pt2 = keypoints2[nn]
            src_points.append(pt1.pt)
            dst_points.append(pt2.pt)

    return src_points, dst_points


house = cv2.imread('../resources/house2.png')
house_gray = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(house_gray, None)

sift_house = np.zeros(house_gray.shape)
sift_house = cv2.drawKeypoints(image=house_gray, keypoints=kp, outImage=sift_house)
cv2.imwrite('sift_keypoints.jpg', sift_house)


img = cv2.imread('../resources/map2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp_img, des_img = sift.detectAndCompute(img_gray, None)

src_points, dst_points = find_pairs(kp_img, des_img, kp, des)
matching_img = np.vstack(
    (
        np.hstack((house, np.zeros(shape=(house.shape[0], img.shape[1], 3)))),
        np.hstack((np.zeros(shape=(img.shape[0], house.shape[1], 3)), img))
    )
)

for i in range(len(src_points)):
    pt1 = src_points[i]
    pt2 = dst_points[i]
    from_ = (int(pt2[0]), int(pt2[1]))
    to_ = int(house.shape[0] + pt1[0]), int(house.shape[1] + pt1[1])
    print(from_, to_)

    pt = (to_[0] - from_[0], to_[1] - from_[1])
    # cv2.line(matching_img, from_, to_, (0, 255, 255))
    cv2.rectangle(matching_img, pt,
        (pt[0] + house.shape[0], pt[1] + house.shape[1]), (0, 0, 255))

cv2.imwrite('matching.jpg', matching_img)


clust = DBSCAN(eps=30, min_samples=4).fit(src_points)
# clust = MeanShift().fit(src_points)

n_clusters_ = len(set(clust.labels_)) - (1 if -1 in clust.labels_ else 0)

clusters = []
print("num clusters = ", n_clusters_)

dst_points = np.array(dst_points)
src_points = np.array(src_points)

for label in range(n_clusters_):
    dst = dst_points[clust.labels_ == label]
    src = src_points[clust.labels_ == label]

    if len(dst) > THRESHOLD_NUM_POINTS:
        res = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        print(label)
        print(res)
        M = res[0]
        mask = res[1]

        if M == None:
            continue
        # print(M)
        # matchesMask = mask.ravel().tolist()

        h,w = house_gray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        src = cv2.perspectiveTransform(pts,M)

        img = cv2.polylines(img,[np.int32(src)],True,255,3, cv2.LINE_AA)

cv2.imwrite('sift_result.jpg', img)
