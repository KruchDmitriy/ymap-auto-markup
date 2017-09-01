from cv2 import imread
from numpy.random import permutation
from os import listdir
from os.path import abspath


class DataManager:
    def __init__(self):
        self.PATH_TO_IMG = abspath("../preprocessed-data/imgs") + '/'
        self.PATH_TO_LABELS = abspath("../preprocessed-data/labels") + '/'
        self.PATH_TO_MODELS = abspath("../data/models/") + '/'
        self.PATH_TO_RESULTS = abspath("../results/") + '/'

        self.TEST_IMAGE_NAME = "3band_AOI_1_RIO_img4599.png"
        self.TEST_IMAGE = self.PATH_TO_IMG + self.TEST_IMAGE_NAME
        self.TEST_LABEL = self.PATH_TO_LABELS + self.TEST_IMAGE_NAME

    def get_test_image(self):
        return imread(self.TEST_IMAGE), imread(self.TEST_LABEL)

    def data_generator(self, shuffle=True):
        files = listdir(self.PATH_TO_IMG)
        if shuffle:
            files = permutation(files)

        for file in files:
            img = imread(self.PATH_TO_IMG + file)
            lab = imread(self.PATH_TO_LABELS + file)
            yield img, lab

    def get_path_to_img(self):
        return self.PATH_TO_IMG

    def get_path_to_labels(self):
        return self.PATH_TO_LABELS

    def get_path_to_models(self):
        return self.PATH_TO_MODELS

    def get_path_to_results(self):
        return self.PATH_TO_RESULTS
