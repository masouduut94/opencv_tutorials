import cv2
import numpy as np
from assets import AssetMeta

if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, None, fx=.5, fy=.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY_INV)

    ad_thresh_mean_c = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12)
    gaus = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 12)

    cv2.imshow("img", img)
    cv2.imshow("thresholded image", threshold)
    cv2.imshow("MEAN_C", ad_thresh_mean_c)
    cv2.imshow("GAUS", gaus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
