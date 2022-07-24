import cv2
import numpy as np

sq = np.zeros((400, 400, 1), np.uint8)
cv2.rectangle(sq, (75, 75), (300, 300), 255, -2)
cv2.imshow("square", sq)
cv2.waitKey(0)

circle = np.zeros((400, 400, 1), np.uint8)
cv2.circle(circle, (300, 300), 75, (255, 0, 0), -1)
cv2.imshow("circle", circle)
cv2.waitKey(0)

cv2.destroyAllWindows()


AND_operation = cv2.bitwise_and(sq, circle)
cv2.imshow("AND operation", AND_operation)
cv2.waitKey()




