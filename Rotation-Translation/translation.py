"""
SRC: https://www.youtube.com/watch?v=-w9mKbtsUyw

"""
import cv2
import numpy as np

from assets import AssetMeta

if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, None, fx=.5, fy=.5)

    h, w, _ = img.shape
    print(f"Size: {w}X{h}")
    print("============")

    h_new, w_new = h/4, w/4

    T = np.float32([[1, 0, h_new], [0, 1, w_new]])
    print(T)

    translation = cv2.warpAffine(img, T, (h, w))

    cv2.imshow("translation", translation)
    cv2.waitKey()


