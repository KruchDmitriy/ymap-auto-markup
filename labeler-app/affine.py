import numpy as np


class AffineMx:
    @staticmethod
    def trans_from_params(shift, theta, scale):
        return np.matmul(
            np.matmul(
                AffineMx.trans(shift[0], shift[1]),
                AffineMx.scale(scale)),
                AffineMx.rotate(theta)
        )

    @staticmethod
    def rotate(theta):
        return np.array([
            [np.cos(theta), np.sin(theta), 0.],
            [-np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]
        ])

    @staticmethod
    def trans(dx, dy):
        return np.array([
            [1., 0., dx],
            [0., 1., dy],
            [0., 0., 1.]
        ])

    @staticmethod
    def scale(sigma):
        return np.array([
            [sigma, 0., 0.],
            [0., sigma, 0.],
            [0., 0., 1.]
        ])

    @staticmethod
    def scale_around(sigma, center):
        scale = np.array([
            [sigma, 0., 0.],
            [0., sigma, 0.],
            [0., 0., 1.]
        ])

        trans_to = AffineMx.trans(-center[0], -center[1])
        trans_from = AffineMx.trans(center[0], center[1])

        return np.matmul(np.matmul(trans_from, scale), trans_to)

    @staticmethod
    def rotate_around(theta, center):
        rotation = np.array([
            [np.cos(theta), np.sin(theta), 0.],
            [-np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]
        ])

        trans_to = AffineMx.trans(-center[0], -center[1])
        trans_from = AffineMx.trans(center[0], center[1])

        return np.matmul(np.matmul(trans_from, rotation), trans_to)

    @staticmethod
    def rotate2d(theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])

    @staticmethod
    def scale2d(sigma):
        return np.array([
            [sigma, 0.],
            [0., sigma],
        ])


class PolyTransform:
    @staticmethod
    def rotate(poly, theta):
        return np.matmul(poly, np.transpose(AffineMx.rotate2d(theta)))

    @staticmethod
    def scale(poly, sigma):
        return np.matmul(poly, np.transpose(AffineMx.scale2d(sigma)))

    @staticmethod
    def translate(poly, dx, dy):
        return poly + np.array((dx, dy))

    @staticmethod
    def translate_vec(poly, shift):
        return poly + shift
