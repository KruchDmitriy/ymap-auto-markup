import numpy as np
from matplotlib import pyplot as plt

precision = 1e-7


def sim(left, right):
    result = 0
    for (a, b) in zip(left, right):
        result += (float(a) - float(b)) * (float(a) - float(b)) / 255 / 255
    return np.exp(-result)


if __name__ == "__main__":
    image = np.zeros(shape=(5, 5, 3), dtype="uint8")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = [0, 0, 0] if i <= j else [255, 255, 255]

    plt.imshow(image)
    plt.show()

    height = image.shape[0]
    width = image.shape[1]
    affinity = np.zeros(shape=(image.shape[0] * image.shape[1], image.shape[0] * image.shape[1]), dtype="float64")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total = 0
            if i > 0:
                s = sim(image[i][j], image[i - 1][j])
                total += s
                affinity[i * width + j][(i - 1) * width + j] = -s
            if i < height - 1:
                s = sim(image[i][j], image[i + 1][j])
                total += s
                affinity[i * width + j][(i + 1) * width + j] = -s
            if j > 0:
                s = sim(image[i][j], image[i][j - 1])
                total += s
                affinity[i * width + j][i * width + j - 1] = -s
            if j < width - 1:
                s = sim(image[i][j], image[i][j + 1])
                total += s
                affinity[i * width + j][i * width + j + 1] = -s
            affinity[i * width + j][i * width + j] = total

    print(affinity)
    sigma, q = np.linalg.eigh(affinity)
    # for i in range(affinity.shape[0]):
    #     buffer = ""
    #     for j in range(affinity.shape[1]):
    #         if np.abs(affinity[i][j]) > precision:
    #             buffer += " %d:%.3g" % (j, affinity[i][j])
    #     if buffer != "":
    #         print "%d: %s" % (i, buffer)
    # print(np.linalg.norm(affinity - q.dot(np.diag(sigma).dot(q.transpose()))), np.linalg.norm(affinity))

    # print(sigma, q)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = 0
            for z in range(sigma.shape[0]):
                if np.abs(sigma[z]) > precision:
                    val += 1 / np.sqrt(sigma[z]) * q[i * width + j][z]
            image[i][j] = [val * 255, val * 255, val * 255]
    plt.imshow(image, interpolation="none")
    plt.show()
