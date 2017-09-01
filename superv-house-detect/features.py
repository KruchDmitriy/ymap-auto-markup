from cv2 import HOGDescriptor

import functools
from descriptors.circle_desc import CircleDescriptor
from numpy import concatenate, zeros


class HoGFeatures:
    def __init__(self, bbox_height=64, bbox_width=64, hog_params=None):
        self.bbox_height = bbox_height
        self.bbox_width = bbox_width

        self.default_hog_params = {
            "win_size": (self.bbox_height, self.bbox_width),
            "block_size": (16, 16),
            "block_stride": (16, 16),
            "cell_size": (8, 8),
            "nbins": 9,
            "deriv_aperture": 3,
            "win_sigma": 4.,
            "histogram_norm_type": 0,
            "L2_hys_threshold": 0.2,
            "gamma_correction": True,
            "nlevels": 64
        }

        if hog_params is None:
            hog_params = self.default_hog_params

        self.win_stride = (8, 8)
        self.padding = (0, 0)

        self.hog = HOGDescriptor(*hog_params.values())
        self.img = None

    def length(self):
        return self.hog.getDescriptorSize() + 5

    def get_bbox_height(self):
        return self.bbox_height

    def get_bbox_width(self):
        return self.bbox_width

    @functools.lru_cache(maxsize=128, typed=False)
    def __hog_compute(self, i, j):
        return self.hog.compute(self.img, self.win_stride, self.padding, locations=((i, j),))

    def set_image(self, img):
        self.img = img

    def apply(self, x, y):
        """
            img     -- image to compute HoG
            (x, y)  -- the point (x -- row, y -- column from top left corner of image)
                       to compute features
        """
        if self.img is None:
            raise Exception("please set image before applying")

        # (i, j)  -- top, left corner of HoG bounding box
        i = x - x % self.bbox_height
        j = y - y % self.bbox_width

        if i + self.bbox_height >= self.img.shape[0]:
            i = self.img.shape[0] - self.bbox_height - 1

        if j + self.bbox_width >= self.img.shape[1]:
            j = self.img.shape[1] - self.bbox_width - 1

        hist = self.__hog_compute(i, j)
        hist = hist.reshape((hist.shape[0],))

        vec = zeros(shape=(self.length(),), dtype='float64')
        vec[:self.hog.getDescriptorSize()] = hist
        vec[-5] = x - i
        vec[-4] = y - j
        vec[-3] = self.img[x][y][0]
        vec[-2] = self.img[x][y][1]
        vec[-1] = self.img[x][y][2]

        return vec


class HoGCirclesFeatures(HoGFeatures):
    def __init__(self, bbox_height=64, bbox_width=64, hog_params=None, circle_desc_params=None):
        super().__init__(bbox_height, bbox_width, hog_params)

        default_circle_desc_params = {
            "radius": 32.,
            "num_circles": 12
        }

        self.circle_desc_params = circle_desc_params

        if circle_desc_params is None:
            self.circle_desc_params = default_circle_desc_params

        self.circle_desc = CircleDescriptor(*self.circle_desc_params.values(), False)

    def length(self):
        return super().length() + self.circle_desc_params["num_circles"] * 6

    def apply(self, x, y):
        vec = super().apply(x, y)

        circles = self.circle_desc.compute(self.img, x, y)
        vec[self.hog.getDescriptorSize():-5] = circles

        return vec
