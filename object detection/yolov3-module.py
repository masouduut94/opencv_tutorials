import cv2
import numpy as np
from time import time

from assets import AssetMeta

meta = AssetMeta()


class YoloDetector:
    def __init__(self, use_tiny=True, use_gpu=False):
        cfg1, w1, n1 = meta.YOLOV3
        cfg2, w2, _ = meta.YOLOV3_TINY

        self.weights = w2 if use_tiny else w1
        self.cfg = cfg2 if use_tiny else cfg1
        self.net = cv2.dnn.readNet(self.weights, self.cfg)

        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = []
        with open(n1, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN

    def detect(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
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

        return boxes, confidences, class_ids

    def process(self, frame, boxes, confidences, class_ids, classes=None):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                if classes is not None:
                    if label not in classes:
                        continue
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)),
                            (x, y + 30), self.font, 1, (255, 255, 255), 2)
        return frame


if __name__ == '__main__':
    vdo = cv2.VideoCapture(0)
    detector = YoloDetector(use_tiny=False, use_gpu=True)

    while True:
        st = time()
        status, frame = vdo.read()
        boxes, confs, class_ids = detector.detect(frame)
        frame = detector.process(frame, boxes, confs, class_ids)
        end = time() - st
        fps = f"FPS: {(1 / end):.2f}"
        cv2.putText(frame, fps, (10, 50), detector.font, 2, (0, 0, 0), 1)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)  # wait 1ms the loop will start again, and we will process the next frame
        if key == ord('q'):  # esc key stops the process
            break
    vdo.release()
    cv2.destroyAllWindows()





