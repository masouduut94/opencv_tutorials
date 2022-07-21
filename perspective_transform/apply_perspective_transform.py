import cv2
import numpy as np

from assets import AssetMeta
from meta import Coordinates as CO, Colors as C

m = AssetMeta()
video_file = m.VIDEO_TT

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_file)
    status, frame = cap.read()

    contour = np.array([CO.tl, CO.bl, CO.br, CO.tr])

    while status:

        # frame = cv2.circle(frame, CO.tl, 10, C.red, -1)
        # frame = cv2.circle(frame, CO.bl, 10, C.green, -1)
        # frame = cv2.circle(frame, CO.tr, 10, C.blue, -1)
        # frame = cv2.circle(frame, CO.br, 10, (0, 255, 255), -1)

        frame = cv2.drawContours(frame, [contour.astype(int)], 0, C.orange, 2)

        pts1 = np.float32([CO.tl, CO.bl, CO.tr, CO.br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        pts1_1 = np.float32([CO.tr, CO.tl, CO.br, CO.bl])
        pts2_2 = np.float32([[0, 0], [0, 640], [480, 0], [480, 640]])

        matrix = cv2.getPerspectiveTransform(pts1_1, pts2_2)
        transformed_image = cv2.warpPerspective(frame, matrix, (480, 640))

        key = cv2.waitKeyEx(30)
        if key == ord('q'):
            break
        cv2.imshow("frame", frame)
        cv2.imshow("TABLE", transformed_image)
        status, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
