import cv2
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('../results/hog_10K.png', 0)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(blur)[0]

    dst = lsd.drawSegments(img, lines)

    plt.axis('off')
    plt.imshow(dst)
    plt.show()


if __name__ == "__main__":
    main()
