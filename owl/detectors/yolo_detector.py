import cv2

from owl.detectors.detector_interface import DetectorInterface


class YoloDetector(DetectorInterface):
    """Yolo Detector"""

    def detect(self, frame, conf_threshold, nms_threshold, model) -> tuple:
        try:
            classes, scores, boxes = model.detect(
                frame=frame,
                confThreshold=conf_threshold,
                nmsThreshold=nms_threshold
            )
            return classes, scores, boxes, None
        except Exception as ex:
            return None, None, None, ex

    def load_classes(self, class_file_path) -> tuple:
        try:
            classes = open(class_file_path).read().strip().split('\n')
            return classes, None
        except Exception as ex:
            return None, ex

    def load_detector_model(self, cfg_file_path, wgt_file_path) -> object:
        try:
            net = cv2.dnn.readNetFromDarknet(cfg_file_path, wgt_file_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(416, 416), scale=1 / 255)
            return model, None
        except Exception as ex:
            return None, ex
