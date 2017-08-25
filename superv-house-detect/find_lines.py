import numpy as np
import cv2
from matplotlib import pyplot as plt


TEST_IMAGE = "hog_10K_small.png"


def calc_coords(alpha, beta, tau_0, tau_1):
    cs = np.cos(alpha)
    sn = np.sin(alpha)

    x_0 = int(tau_0 * cs + beta * sn)
    y_0 = int(tau_0 * sn - beta * cs)

    x_1 = int(tau_1 * cs + beta * sn)
    y_1 = int(tau_1 * sn - beta * cs)

    print(x_0, y_0, x_1, y_1)

    return x_0, y_0, x_1, y_1


def find_edges(img, num_instances=100):
    height = img.shape[0]
    width = img.shape[1]

    side = max(width, height)
    max_beta_value = np.sqrt(2.) / 2. * side

    alphas = np.linspace(0., 2. * np.pi, num_instances)
    betas = np.linspace(-max_beta_value, max_beta_value, side)

    tau_0 = np.linspace(-max_beta_value, max_beta_value, side)
    tau_1 = np.linspace(-max_beta_value, max_beta_value, side)

    # a = np.array([np.cos(alphas), np.sin(alphas)])
    # b = np.array([np.cos(np.pi / 2. + alpha) * beta, np.sin(np.pi / 2. + alphas) * beta])
    # a, b, t0, t1 = np.meshgrid([alphas, betas, tau_0, tau_1])
    values = []
    max_value = -np.inf
    args = None

    for i in range(num_instances):
        for j in range(side):
            for k in range(side):
                for l in range(side):
                    value = target_function(img, alphas[i], betas[j], tau_0[k], tau_1[l])
                    if value > max_value:
                        max_value = value
                        args = (alphas[i], betas[j], tau_0[k], tau_1[l])
                        print(args)

                        x_0, y_0, x_1, y_1 = calc_coords(*args)
                        copy_img = img.copy()
                        copy_img = cv2.normalize(copy_img, copy_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
                        copy_img = cv2.line(copy_img, (x_0, y_0), (x_1, y_1), (255, 0, 0), 1)
                        print(copy_img.shape)
                        plt.imshow(copy_img)
                        plt.show()

                    values.append(value)

    plt.hist(values)
    plt.show()



def target_function(img, alpha, beta, tau_0, tau_1):
    result = None
    a = np.array([np.cos(alpha), np.sin(alpha)])
    b = np.array([np.sin(alpha), -np.cos(alpha)])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = np.array([i, j])
            if (np.dot(a, x) - tau_0) < 0 or (tau_1 - np.dot(a, x)) < 0:
                continue

            p_x = np.log(img[i][j][0])

            inv_x = x - 2. * (np.dot(b, x) - beta) * b
            inv_i = int(inv_x[0] + 0.5)
            inv_j = int(inv_x[1] + 0.5)

            p_inv_x = np.log(img[inv_i][inv_j][0]) if inv_i < img.shape[0] and inv_i >= 0\
                                                    and inv_j < img.shape[1] and inv_j >= 0 else 0.

            value = (np.dot(b, x) - beta) * (np.dot(b, x) - beta) * (p_x - p_inv_x)
            if result == None:
                result = value
            else:
                result += value

    if result == None:
        return -np.inf

    return result


def main():
    img = cv2.imread(TEST_IMAGE)

    dst = np.zeros(shape=img.shape)
    dst = cv2.normalize(img, dst, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    find_edges(dst)


if __name__ == "__main__":
    main()
