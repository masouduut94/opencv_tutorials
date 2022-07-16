"""
https://www.delftstack.com/howto/python/opencv-kalman-filter/

"""
import cv2
import numpy as np

measured = []
predicted = []
dr_frame = np.zeros((400, 400, 3), np.uint8)
mp = np.array((2, 1), np.float32)
tp = np.zeros((2, 1), np.float32)


def on_mouse(k, x, y, s, p):
    global mp, measured
    mp = np.array([[np.float32(x)], [np.float32(y)]])
    measured.append((x, y))


def paint_canvas():
    global dr_frame, measured, predicted
    for i in range(len(measured) - 1):
        cv2.line(dr_frame, measured[i], measured[i + 1], (0, 100, 0))
    for i in range(len(predicted) - 1):
        cv2.line(dr_frame, predicted[i], predicted[i + 1], (0, 0, 200))


def reset_canvas():
    global measured, predicted, dr_frame
    measured = []
    predicted = []
    dr_frame = np.zeros((400, 400, 3), np.uint8)


cv2.namedWindow("Sample")
cv2.setMouseCallback("Sample", on_mouse)
kalman_fil = cv2.KalmanFilter(4, 2)
kalman_fil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman_fil.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman_fil.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

while True:
    kalman_fil.correct(mp)
    tp = kalman_fil.predict()
    predicted.append((int(tp[0]), int(tp[1])))
    paint_canvas()
    cv2.imshow("Output", dr_frame)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        break
    if k == ord('r'):
        reset_canvas()
