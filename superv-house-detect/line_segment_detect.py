import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('hog_10K.png', 0)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(blur)[0]

    dst = lsd.drawSegments(img, lines)

    plt.axis('off')
    plt.imshow(dst)
    plt.show()

    cv2.imwrite('detected_edges.png', dst)


if __name__ == "__main__":
    main()
