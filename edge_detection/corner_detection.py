"""
Source code from YouTube: Tech with Tim channel
https://www.youtube.com/watch?v=I7lCpTOfxF4

"""
import random
from pathlib import Path
import numpy as np
import cv2

path1 = Path("../assets/IMAGES/chess_board.PNG")
path2 = Path("../assets/IMAGES/chess_board_side.jpg")

img = cv2.imread(path1.as_posix())
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

N_corners_to_track = 50

"""
# maxCorners: how many corners
# qualityLevel: minimum confidence
# minDistance: Euclidean distance between 2 corners must be greater than this to include that corner.
"""
corners = cv2.goodFeaturesToTrack(gray,
                                  maxCorners=N_corners_to_track,
                                  qualityLevel=0.01,
                                  minDistance=10)

corners = np.int32(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 10, (255, 0, 255), 3)

# ####### Just for fun!!! ####
# ####### Just for fun!!! ####
# for i, _ in enumerate(corners):
#     for j in range(i+1, len(corners)):
#
#         corner1 = tuple(corners[i][0])
#         corner2 = tuple(corners[j][0])
#
#         color = tuple([random.randint(0, 255) for _ in range(3)])
#         cv2.line(img, corner1, corner2, color, 1)


cv2.imshow('FRAME', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
