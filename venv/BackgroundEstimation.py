import numpy as np

class BackgroundEstimator:
    def __init__(self):
        self.histWeight = 0.5
        self.devWeight = 0.2

    def initialise(self, firstImg):
        self.avImg = np.asarray(firstImg)

    def estimateBackground(self, img):
        newWeight = 1 - self.histWeight

        self.avImg = newWeight * np.asarray(img) + self.histWeight * self.avImg

        delta = pow(self.avImg - img,  2) * self.devWeight

        resImg = self.avImg
        resImg = img[np.where(img[:,:] < self.avImg[:,:] + delta[:,:])]
        resImg = img[np.where(img[:,:] > self.avImg[:,:] - delta[:,:])]

        return resImg.astype(np.uint8)
