"""
Resources:
- https://pjreddie.com/darknet/yolo/
- https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

SetUp for GPU inference:
- https://learnopencv.com/how-to-use-opencv-dnn-module-with-nvidia-gpu-on-windows/


"""

import cv2
import numpy as np
import time

# Load YOLO
from assets import AssetMeta

meta = AssetMeta()

weight, cfg, names_txt = meta.YOLOV3

net = cv2.dnn.readNet(weight, cfg)  # Original yolov3

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open(names_txt, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
# print(layer_names)
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)  # 0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN

frame_id = 0

while True:
    st = time.time()
    _, frame = cap.read()
    height, width, channels = frame.shape
    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

    net.setInput(blob)
    outs = net.forward(output_layers)
    # print(outs[1])

    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x, y, w, h])  # put all rectangle areas
                confidences.append(
                    float(confidence))  # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

    elapsed_time = time.time() - st
    fps = f"FPS: {(1 / elapsed_time):.2f}"
    cv2.putText(frame, fps, (10, 50), font, 2, (0, 0, 0), 1)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)  # wait 1ms the loop will start again, and we will process the next frame

    if key == ord('q'):  # esc key stops the process
        break

cap.release()
cv2.destroyAllWindows()
