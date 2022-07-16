import cv2
import numpy as np


class Meta:
    CAR_XML = '../assets/haarcascade-xml/cars.xml'
    CAR_VIDEO = '../assets/VIDEOS/cars.avi'
    PEDESTRIAN_XML = '../assets/haarcascade-xml/pedestrian.xml'
    PEDESTRIAN_VIDEO = '../assets/VIDEOS/pedestrians.avi'


cap = cv2.VideoCapture(Meta.PEDESTRIAN_VIDEO)
car_cascade = cv2.CascadeClassifier(Meta.PEDESTRIAN_XML)

status, frame = cap.read()

while status:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car = car_cascade.detectMultiScale(gray, 1.3, 2)
    for (a, b, c, d) in car:
        frame = cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 210), 4)

    cv2.imshow('video', frame)
    if cv2.waitKey(25) == ord('q'):
        break

    status, frame = cap.read()

cv2.destroyAllWindows()
