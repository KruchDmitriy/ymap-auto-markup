import os
import collections
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

from torch.utils import data
from sklearn.model_selection import train_test_split


class spacenetLoader(data.Dataset):
    def __init__(self, img_rows, img_cols, split="train"):
        sys.path.insert(0, '..')
        from utils import DataManager

        self.data_manager = DataManager()

        self.root = self.data_manager.get_path_to_img()
        self.split = split
        self.img_size = [img_rows, img_cols]
        # self.mean = np.array([104.00699, 116.66877, 122.67892]) seems to be RGB
        self.mean = np.array([122.67892, 116.66877, 104.00699])
        self.n_classes = 2
        self.files = collections.defaultdict(list)

        files = np.random.permutation(os.listdir(self.root))
        num_files = len(files)
        train, test, val = np.split(files, [int(num_files * 0.8), int(num_files * 0.9)])
        print(len(train), len(test), len(val))

        self.files["train"] = train
        self.files["test"] = test
        self.files["val"] = val

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.data_manager.get_path_to_img() + img_name
        lbl_path = self.data_manager.get_path_to_labels() + img_name

        img = cv2.imread(img_path)
        lab = cv2.imread(lbl_path, 0)

        img, lab = self.transform(img, lab)

        return img, lab

    def transform(self, img, lab):
        img = img.astype(np.float64)

        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        lab = cv2.resize(lab, (self.img_size[0], self.img_size[1]))

        img -= self.mean
        img /= 255.0
        img = img.transpose(2, 0, 1)
        lab = lab.reshape((1, lab.shape[0], lab.shape[1]))
        lab[lab != 0] = 1

        img = torch.from_numpy(img).float()
        lab = torch.from_numpy(lab).long()

        return img, lab

    def get_test_item(self):
        img, lab = self.data_manager.get_test_image()
        img, lab = self.transform(img, lab)
        return img, lab
