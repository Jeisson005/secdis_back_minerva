import cv2
import numpy as np
from owl.detectors.detector_interface import DetectorInterface


class TensorflowDetector(DetectorInterface):
    """Tensorflow Detector"""

    def detect(self, frame, conf_threshold, nms_threshold, model) -> tuple:
        try:
            classes = []
            scores = []
            boxes = []
            rows = frame.shape[0]
            cols = frame.shape[1]
            model.setInput(
                cv2.dnn.blobFromImage(
                    frame, size=(300, 300),
                    swapRB=True, crop=False
                )
            )
            cv_out = model.forward()

            for detection in cv_out[0, 0, :, :]:
                score = float(detection[2])
                if score > 0.5:
                    left = int(detection[3] * cols)
                    top = int(detection[4] * rows)
                    right = int(detection[5] * cols)
                    bottom = int(detection[6] * rows)

                    x = left
                    y = top
                    w = abs(right - left)
                    h = abs(bottom - top)

                    scores.append([score])
                    boxes.append([x, y, w, h])
                    classes.append([int(detection[1])])

            classes = np.array(classes)
            scores = np.array(scores)
            boxes = np.array(boxes)

            return classes, scores, boxes
        except Exception as ex:
            return None, ex, None

    def load_classes(self, class_file_path) -> tuple:
        classes = ["background", "person", "bicycle", "car", "motorcycle",
                   "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                   "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                   "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                   "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                   "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                   "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                   "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                   "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                   "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                   "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                   "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        return classes, None

    def load_detector_model(self, cfg_file_path, wgt_file_path) -> object:
        try:
            net = cv2.dnn.readNetFromTensorflow(wgt_file_path, cfg_file_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            return net, None
        except Exception as ex:
            return None, ex
