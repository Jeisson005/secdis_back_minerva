"""Detector Interface"""
import abc


class DetectorInterface(metaclass=abc.ABCMeta):
    """Detector Interface"""

    @abc.abstractmethod
    def detect(self, frame, conf_threshold, nms_threshold, model) -> tuple:
        """Detect objects in a frame

        :param frame: The input frame.
        :param conf_threshold: A threshold used to filter boxes by
        confidences.
        :type conf_threshold: float
        :param nms_threshold: A threshold used in non maximum
        suppression.
        :type nms_threshold: float
        :param model: Detector model

        :return: A tuple with (class_ids, scores, boxes) where
        class_ids are class indexes in result detection, scores are a
        set of corresponding confidences and boxes are a set of
        bounding boxes (x, y, w, h).
        """

    @abc.abstractmethod
    def load_classes(self, class_file_path) -> tuple:
        """Load and return the names of the classes or labels that
        model is able to detect.

        :param class_file_path: class file path
        :type class_file_path: str

        :return: list of the names of the classes or labels and error
        (if exists)
        """

    @abc.abstractmethod
    def load_detector_model(self, cfg_file_path, wgt_file_path) -> object:
        """Load detector model

        :param cfg_file_path: configuration file path
        :type cfg_file_path: str
        :param wgt_file_path: weight file path
        :type wgt_file_path: str

        :return: Detector model and error (if exists)
        """
