import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

w = int(cap.get(3))
h = int(cap.get(4))

# Loading image

while True:
    status, image = cap.read()

    result = face_mesh.process(image)
    if result.multi_face_landmarks is not None:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt = facial_landmarks.landmark[i]
                x = int(pt.x*w)
                y = int(pt.y*h)
                image = cv2.circle(image, (x, y), 3, (100, 100, 0), -1, )

    cv2.imshow("image", image)
    k = cv2.waitKey(20)
    if k == ord('q'):
        break

