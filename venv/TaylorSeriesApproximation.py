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
        origimg = cv2.rectangle(img, a, b, c, -1)
        origimg = cv2.GaussianBlur(origimg, (5, 5), 0)

        cv2.imshow("TaylorSeriesWin", origimg)
        cv2.waitKey(1000000000)

        img = self.findXDerivation(origimg)

        cv2.imshow("TaylorSeriesWin", img)
        cv2.waitKey(1000000000)

        img = self.findYDerivation(img)

        # Show the result
        cv2.imshow("TaylorSeriesWin", img)

        # Wait for a key to be pressed before closing the window
        cv2.waitKey(1000000000)

    def findXDerivation(self, img):
        derivImg = np.zeros((img.shape[0], img.shape[1]-1, img.shape[2]), np.float)

        derivImg[:,:,:] = img[:, :-1, :] - img[:, 1:, :]

        return abs(derivImg)

    def findYDerivation(self, img):
        derivImg = np.zeros((img.shape[0]-1, img.shape[1], img.shape[2]), np.float)

        derivImg[:,:,:] = img[:-1, :, :] - img[1:, :, :]

        return abs(derivImg)