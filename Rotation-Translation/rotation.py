"""
SRC: https://www.youtube.com/watch?v=yDGWSeI0POw&list=PLlH6o4fAIji6epixHdlBYZthV9YVEfT9P&index=8

"""

import cv2
import numpy as np

from assets import AssetMeta

if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, None, fx=.5, fy=.5)

    simple_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("original", img)
    cv2.imshow("simple rotate", simple_rot)

    h, w, _ = img.shape

    rot_Mat = cv2.getRotationMatrix2D((w/2, h/2), 125, 0.5)
    rot_image_output = cv2.warpAffine(img, rot_Mat, (w, h))
    cv2.imshow("rotated", rot_image_output)

    cv2.waitKey()
    cv2.destroyAllWindows()
