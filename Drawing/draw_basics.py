import cv2
import numpy as np
from pathlib import Path
from os.path import isfile
from assets import AssetMeta


if __name__ == '__main__':
    meta = AssetMeta()
    path = meta.VIDEO_TT

    # assert isfile(path)
    cap = cv2.VideoCapture(path)
    # assert cap.isOpened(), " Can not read the video"

    status, frame = cap.read()
    print(status)
    while status:
        frame = cv2.resize(frame, None, fx=.5, fy=.5)
        width, height, _ = frame.shape

        # Drawing Line:
        frame = cv2.line(frame, pt1=(10, 30), pt2=(width, height), color=(255, 0, 0), thickness=2)

        # Drawing a rectangle:
        cv2.rectangle(frame, pt1=(50, 50), pt2=(200, 200), color=(0, 255, 0), thickness=7)

        # Drawing a circle
        cv2.circle(frame, center=(300, 300), radius=10, color=(0, 0, 255), thickness=4)

        # Drawing Texts
        cv2.putText(frame, "Hello classroom", (50, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 3)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

        status, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
