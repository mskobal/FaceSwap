""" Generate image from array of numbers. """


import numpy as np
import cv2


def main():
    image = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    image = image * 255.
    diamond = cv2.resize(image, (90, 180), interpolation=cv2.INTER_AREA)
    cv2.imwrite("diamond.png", diamond)


if __name__ == "__main__":
    main()
