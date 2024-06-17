"""Bird's-Eye View"""
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley",
               "Deepak Birla"]

import cv2


class BirdsEyeView:
    """ Birds Eye View"""

    @staticmethod
    def scale_width_and_height(vc):
        """ Get scale, function gives scale for birds eye view

            :param vc:

            :return:
        """
        # Get video height and width
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))

        new_w = 400
        new_h = 600
        return float(new_w / width), float(new_h / height)
