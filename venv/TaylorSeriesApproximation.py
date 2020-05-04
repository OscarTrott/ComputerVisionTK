import cv2
import numpy as np


class TaylorSeriesApproximator:
    """Class to implement taylor series approximation of 1 and 2 dimensional functions"""

    def __init__(self):
        self.initialised = True

        self.imageSize = (512, 512, 3)


    def start(self):
        img = np.zeros(self.imageSize, np.uint8)

        cv2.imshow("TaylorSeriesWin", img)

        # Wait for a key to be pressed before closing the window
        cv2.waitKey(1000000000)