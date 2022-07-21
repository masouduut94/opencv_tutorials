import cv2
import numpy as np


def get_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        points.append([x, y])
        print(points)
        if len(points) == 4:
            get_perspective()
    cv2.imshow("image", img)


def get_perspective():
    width = points[1][0] - points[0][0]
    height = points[2][1] - points[0][1]

    points1 = np.float32(points[0:4])
    points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    M = cv2.getPerspectiveTransform(points1, points2)
    output = cv2.perspectiveTransform(img, M, (width, height))
    cv2.imshow("OUTPUT", output)


if __name__ == '__main__':
    points = []
    img = cv2.imread("../assets/IMAGES/book.jpeg")

    while True:
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', get_points)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        elif cv2.waitKey(1) & 0xff == ord('r'):
            img = cv2.imread("../assets/IMAGES/book.jpeg")
            points = []
            cv2.imshow('image', img)
            cv2.setMouseCallback('image', get_points)
        else:
            continue

    cv2.destroyAllWindows()
