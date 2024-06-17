from owl.detectors.detector_interface import DetectorInterface
from owl.detectors.yolo_detector import YoloDetector
from owl.detectors.tensorflow_detector import TensorflowDetector


class DetectorFactory:
    """Detector Factory"""

    @staticmethod
    def get_detector(name) -> DetectorInterface:
        """ Get detector by name

        :param name: detector name
        :type name: str

        :raises ValueError: Detector not found

        :return: detector
        """
        if name == "YOLO":
            return YoloDetector()
        elif name == "TensorFlow":
            return TensorflowDetector()
        else:
            raise ValueError(name)
