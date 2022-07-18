"""
mp_face_det = mp.solutions.face_detection
mp.solutions.pose
mp.solutions.hands
mp.solutions.face_mesh_connections
mp.solutions.objectron
mp.solutions.selfie_segmentation
mp.solutions.hands_connections
mp.solutions.holistic
"""

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh()

# Loading image
image = cv2.imread('../assets/IMAGES/tom/tom1.jpeg')
h, w, _ = image.shape

result = face_mesh.process(image)
for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x*w)
        y = int(pt.y*h)
        image = cv2.circle(image, (x, y), 3, (100, 100, 0), -1, )

cv2.imshow("image", image)
cv2.waitKey(0)

