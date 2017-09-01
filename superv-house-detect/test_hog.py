import numpy as np
import cv2
import argparse

from features import HoGFeatures, HoGCirclesFeatures
from cpp_ottree.tree import Ensemble
from normalization import norm_grad, norm_max
from utils import DataManager


def iou(src, lab):
    src = (src == 255)
    lab = (lab == 255)

    intersection = np.count_nonzero(np.logical_and(src, lab))
    union = np.count_nonzero(np.logical_or(src, lab))

    return intersection / union


def main(args):
    data_manager = DataManager()

    if args.iou:
        if args.img is None:
            print('please specify img and label param')
            return 0

        img = cv2.imread(data_manager.get_path_to_img() + args.img)

        one_channel = img[:, :, 0]
        thresh, dst = cv2.threshold(one_channel, args.thresh, 255, cv2.THRESH_BINARY)

        label = cv2.imread(data_manager.get_path_to_labels() + args.lbl)
        lab_one_channel = label[:, :, 0]

        dst_ = np.zeros(lab_one_channel.shape)
        dst_[:dst.shape[0], :dst.shape[1]] = dst

        print("IoU = ", iou(dst_, lab_one_channel))
        cv2.imwrite(data_manager.get_path_to_results() + args.result, dst)

        return 0

    ensemble = Ensemble(data_manager.get_path_to_models() + args.model, 6, args.verbose)

    bbox_width = 64
    bbox_height = 64
    bbox_half_width = bbox_width // 2
    bbox_half_height = bbox_height // 2

    if args.descr_type == 'hog':
        descriptor = HoGFeatures(bbox_height, bbox_width)
    else:
        descriptor = HoGCirclesFeatures(bbox_height, bbox_width)

    img, label = data_manager.get_test_image()

    if args.img:
        img = cv2.imread(data_manager.get_path_to_img() + args.img)

    probas = np.zeros(shape=(img.shape[0] // bbox_half_height,
                             img.shape[1] // bbox_half_width,
                             bbox_height, bbox_width))

    descriptor.set_image(img)

    for i in np.arange(0, img.shape[0] - bbox_height, bbox_half_height):
        for j in np.arange(0, img.shape[1] - bbox_width, bbox_half_width):
            print(i, j)

            for x in range(bbox_width):
                for y in range(bbox_height):
                    vec = descriptor.apply(int(i) + x, int(j) + y)

                    prob = 1. / (1. + np.exp(-ensemble.predict(vec)))
                    probas[i // bbox_half_height][j // bbox_half_width][x][y] = prob

    if args.normalize == 'grad_desc':
        dst = norm_grad(probas, bbox_height, bbox_width)
    else:
        dst = norm_max(probas, bbox_height, bbox_width, img_height=img.shape[0], img_width=img.shape[1])

    cv2.imwrite(data_manager.get_path_to_results() + args.result, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descr_type', required=True, choices=['hog', 'circles'],
                        help='type of descriptor')
    parser.add_argument('--model', required=True, help='model file name')
    parser.add_argument('--result', required=True, help='name for result image')

    parser.add_argument('--iou', action='store_true', required=False, help='choose when need to compute iou metric')
    parser.add_argument('--thresh', type=int, required=False, help='required! when need to compute iou metric')
    parser.add_argument('--img', required=False, help='image to test')
    parser.add_argument('--lbl', required=False, help='label to test')
    parser.add_argument('--normalize', required=False, default='grad_desc',
                        choices=['max', 'grad_desc'],
                        help='specifies the type of probas merging')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', help='verbose mode')
    args = parser.parse_args()

    main(args)
