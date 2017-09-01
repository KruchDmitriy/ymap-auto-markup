import argparse
from random import randint
from features import HoGFeatures, HoGCirclesFeatures
from utils import DataManager


def main(args):
    data_manager = DataManager()

    bbox_height = 64
    bbox_width = 64
    if args.descr_type == 'hog':
        descriptor = HoGFeatures(bbox_height, bbox_width)
    else:
        descriptor = HoGCirclesFeatures(bbox_height, bbox_width)

    with open(data_manager.get_path_to_models() + args.result, 'w') as feat:
        n_samples = 0
        for img, label in data_manager.data_generator(shuffle=True):
            descriptor.set_image(img)

            for count in range(args.samples_per_img):
                x = randint(0, img.shape[0] - 1)
                y = randint(0, img.shape[1] - 1)

                vec = descriptor.apply(x, y)
                lab = int((label[x][y] == [255, 255, 255])[0])

                # str_id = "%s-(%d,%d)" % (file, x, y)
                feat.write(str(n_samples) + '\t' + str(lab) + '\t' + str(n_samples) + '\t0\t')
                vec.tofile(feat, sep='\t')
                feat.write('\n')

                n_samples += 1
                if args.v and n_samples % 1000 == 0:
                    print(n_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descr_type', required=True, choices=['hog', 'circles'],
                        help='type of descriptor')
    parser.add_argument('-o', '--result', required=True, help='name of result file for dataset')
    parser.add_argument('-s', '--samples_per_img', required=True, type=int,
                        help='number of samples per image')
    parser.add_argument('-v', required=False, action='store_true', default=True, help='verbose mode')

    args = parser.parse_args()

    main(args)
