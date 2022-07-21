import cv2
import mediapipe as mp
import numpy as np
import keyboard
from assets import AssetMeta

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# IMAGE_FILES = ['EM.jpg']
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        # image = cv2.imread("JB.jpg")
        meta = AssetMeta()
        imgFront = cv2.imread(meta.IMG_FACE_ON[1], cv2.IMREAD_UNCHANGED)
        s_h, s_w, _ = imgFront.shape

        imageHeight, imageWidth, _ = image.shape
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if results.detections:
            for detection in results.detections:
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    imageWidth,
                    imageHeight
                )
                Nose_tip_x = pixelCoordinatesLandmark[0]  # NOSE
                Nose_tip_y = pixelCoordinatesLandmark[1]
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    imageWidth,
                    imageHeight
                )
                Left_Ear_x = pixelCoordinatesLandmark[0]  # LEFT EAR
                Left_Ear_y = pixelCoordinatesLandmark[1]
                normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                     mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    imageWidth,
                    imageHeight
                )
                Right_Ear_x = pixelCoordinatesLandmark[0]  # RIGHT EAR
                Right_Ear_y = pixelCoordinatesLandmark[1]
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                    normalizedLandmark.x,
                    normalizedLandmark.y, imageWidth,
                    imageHeight
                )
                Left_EYE_x = pixelCoordinatesLandmark[0]  # LEFT EYE
                Left_EYE_y = pixelCoordinatesLandmark[1]
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                    normalizedLandmark.x,
                    normalizedLandmark.y, imageWidth,
                    imageHeight
                )
                Right_EYE_x = pixelCoordinatesLandmark[0]  # RIGHT EYE
                Right_EYE_y = pixelCoordinatesLandmark[1]
                sunglass_width = Left_Ear_x - Right_Ear_x + 60
                sunglass_height = int((s_h / s_w) * sunglass_width)
                # print('size : ',sunglass_width,sunglass_height)

                imgFront = cv2.resize(imgFront, (sunglass_width, sunglass_height), None, 0.3, 0.3)

                hf, wf, cf = imgFront.shape
                hb, wb, cb = image.shape

                # 194 > 100
                # 80 > 60    160/80 * 60
                y_adjust = int((sunglass_height / 80) * 80)  # adjust value to fine tune
                x_adjust = int((sunglass_width / 194) * 100)

                pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]
                # print(hb-hf)
                # imgResult = cvzone.overlayPNG(imgBack, imgFront, [0, hb-hf])

                hf, wf, cf = imgFront.shape
                hb, wb, cb = image.shape
                *_, mask = cv2.split(imgFront)
                maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
                imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

                imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
                imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
                imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
                maskBGRInv = cv2.bitwise_not(maskBGR)
                imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

                image = cv2.bitwise_and(image, imgMaskFull2)
                image = cv2.bitwise_or(image, imgMaskFull)
                # cv2.namedWindow("Sunglass Effect",cv2.WINDOW_NORMAL)
                # image = cv2.resize(image,(540,960))
        cv2.imshow('Sunglass Effect', image)
        # result.write(image)  Write to a file

        if keyboard.is_pressed('q'):
            break

        cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()