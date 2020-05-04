import cv2
import numpy as np


class TaylorSeriesApproximator:
    """Class to implement taylor series approximation of 1 and 2 dimensional functions"""

    def __init__(self):
        self.initialised = True

        self.imageSize = np.array([512, 512, 3])


    def start(self):
        # Create initial image
        img = np.zeros(self.imageSize, np.uint8)

        rectangleProportion = 0.5

        # Rectangle parameters
        rec1Size = np.rint(self.imageSize[0:2] * rectangleProportion)
        rec1Position1 = np.rint((self.imageSize[0:2] - rec1Size) / 2)
        rec1Position2 = rec1Size + rec1Position1
        rec1Colour = np.array([255, 255, 255], np.uint8)

        a = tuple(rec1Position1.astype(int).tolist())
        b = tuple(rec1Position2.astype(int).tolist())
        c = rec1Colour.tolist()

        # Draw a rectangle
        img = cv2.rectangle(img, a, b, c, 3)

        # Show the result
        cv2.imshow("TaylorSeriesWin", img)

        # Wait for a key to be pressed before closing the window
        cv2.waitKey(1000000000)