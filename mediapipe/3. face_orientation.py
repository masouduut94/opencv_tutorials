import cv2
import mediapipe as mp
import numpy as np
from time import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    status, frame = cap.read()

    st = time()
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # image.flags.writable = False

    results = face_mesh.process(frame)

    # image.flags.writable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * w, lm.y * h)
                        nose_3d = (lm.x * w, lm.y * h, lm.z * 3000)

                    x, y = int(lm.x * w), int(lm.y * h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * w

            cam_matrix = np.array([[focal_length, 0, h/2],
                                   [0, focal_length, h/2],
                                   [0, 0, 1]])

            distance_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distance_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0]*360
            y = angles[1]*360
            z = angles[2]*360

            if y > -10:
                text = 'Looking Left'
            elif y < 10:
                text = "looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, distance_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.putText(image, text, (20, 50), font, 2, (0, 255, 0), 2)
            cv2.putText(image, "X: "+str(np.round(x, 2)), (500, 50), font, 2, (0, 255, 0), 2)
            cv2.putText(image, "Y: "+str(np.round(y, 2)), (500, 100), font, 2, (0, 255, 0), 2)
            cv2.putText(image, "Z: "+str(np.round(z, 2)), (500, 150), font, 2, (0, 255, 0), 2)

        end = time()
        tot = end - st
        fps = 1/tot

        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), font, 1.5, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)

    cv2.imshow("Head Pose Estimation", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()





