import numpy as np
from random import randint
from scipy.spatial import distance
from utils import DataManager

BBOX_WIDTH = 21
BBOX_HEIGHT = 21

BBOX_HALF_WIDTH = BBOX_WIDTH // 2
BBOX_HALF_HEIGHT = BBOX_HEIGHT // 2


def main():
    center_x = BBOX_HALF_HEIGHT
    center_y = BBOX_HALF_WIDTH

    data_manager = DataManager()

    idx = 0

    with open('features_test.txt', 'w') as feat:
        for img, label in data_manager.data_generator(shuffle=True):
            for count in range(1000):
                i = randint(BBOX_HALF_HEIGHT, img.shape[0] - 1 - BBOX_HALF_HEIGHT)
                j = randint(BBOX_HALF_WIDTH, img.shape[1] - 1 - BBOX_HALF_WIDTH)

                roi = img[i - BBOX_HALF_HEIGHT: i + BBOX_HALF_HEIGHT + 1,
                          j - BBOX_HALF_WIDTH: j + BBOX_HALF_WIDTH + 1]

                vec = np.zeros(shape=(BBOX_WIDTH * BBOX_HEIGHT + 3))

                for x in range(BBOX_HEIGHT):
                    for y in range(BBOX_WIDTH):
                        if x == center_x and y == center_y:
                            continue
                        vec[x * BBOX_WIDTH + y] = \
                            distance.euclidean(roi[x][y], roi[center_x][center_y])

                vec[-3] = roi[center_x][center_y][0]
                vec[-2] = roi[center_x][center_y][1]
                vec[-1] = roi[center_x][center_y][2]

                lab = int((label[i][j] == [255, 255, 255])[0])
                str_id = str(idx)

                feat.write(str_id + '\t' + str(lab) + '\t' + str_id + '\t0\t')
                vec.tofile(feat, sep='\t')
                feat.write('\n')

                idx += 1


if __name__ == "__main__":
    main()
