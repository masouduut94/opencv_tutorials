import cv2
import sys
# sys.path.append('..\\assets')
from assets import AssetMeta
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == '__main__':
    meta = AssetMeta()
    img1, img2 = meta.IMG_NORMAL

    img = cv2.imread(img1)
    img = cv2.resize(img, None, fx=.5, fy=.5)
    img_new = adjust_gamma(img, 0.1)
    img_new = cv2.resize(img_new, None, fx=.5, fy=.5)

    cv2.imshow('ORIGINAL', img)
    cv2.imshow("Gamma_COrrection", img_new)
    k = cv2.waitKey(0)
