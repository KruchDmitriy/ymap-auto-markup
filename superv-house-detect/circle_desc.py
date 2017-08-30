import numpy as np
from matplotlib import pyplot as plt


class CircleDescriptor:
    def __init__(self, radius, num_circles, num_channels=3, LOG=False):
        self.radius = radius
        self.num_circles = num_circles
        self.stride = radius / num_circles
        self.moment_funcs = [ np.mean, np.var ]
        self.num_channels = num_channels
        self.LOG = LOG
        self.__create_rings()

    def calc(self, img, x, y):
        R = int(np.round(self.radius))

        roi = img[x - R: x + R + 1, y - R: y + R + 1]
        moments = []

        for ring in self.rings:
            points = roi[ring]

            for func in self.moment_funcs:
                moments.append(np.apply_along_axis(func, 0, points))

        return np.concatenate(moments)

    def __create_rings(self):
        R = int(np.round(self.radius))
        stride = self.radius / self.num_circles

        if self.LOG:
            print("radius " + str(R))
            print("stride " + str(stride))
            print("num_circles " + str(self.num_circles))

        self.rings = np.zeros((2 * R + 1, 2 * R + 1))

        for i in np.arange(-R, R + 1):
            for j in np.arange(-R, R + 1):
                ring_idx = int(np.round(np.sqrt(i * i + j * j) / stride))

                if ring_idx >= self.num_circles:
                    self.rings[i + R, j + R] = -1
                    continue

                self.rings[i + R, j + R] = ring_idx
        res = []

        if self.LOG:
            plt.imshow(self.rings)
            plt.show()

        for i in range(self.num_circles):
            res.append(self.rings == i)

        self.rings = np.array(res)

    def get_descriptor_size(self):
        return self.num_circles * len(self.moment_funcs) * self.num_channels


def main():
    # smoke test
    TEST_IMG = np.random.normal(0, 1, 65 * 65 * 3).reshape((65, 65, 3))
    circ = CircleDescriptor(32, 12, LOG=True)

    vec = circ.calc(TEST_IMG, 32, 32)
    print(vec)

if __name__ == "__main__":
    main()
