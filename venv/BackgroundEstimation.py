import numpy as np
import cv2

class BackgroundEstimator:
    """ Background estimator class
        builds up a statistical representation of the background and allows for it to be removed from captured frames"""
    def __init__(self):
        self.histWeight = 0.99
        self.devWeight = 0.95

        self.sigmas = 1

    def initialise(self, firstImg):
        self.avImg = np.asarray(firstImg)
        self.delta = np.ones_like(self.avImg)

    def estimateBackground(self, img):
        newHistWeight = 1 - self.histWeight
        newDevWeight = 1 - self.devWeight

        imgArray = np.asarray(img)

        self.avImg = newHistWeight * imgArray + self.histWeight * self.avImg
        self.avImg = self.avImg.astype(np.uint8)

        self.delta = pow(self.avImg - imgArray,  2) * newDevWeight + self.devWeight * self.delta
        self.delta = self.delta.astype(np.uint8)

        resImg = self.avImg
        #resImg = np.zeros_like(self.avImg)
        resImg = np.where(imgArray > self.avImg + self.sigmas * self.delta, imgArray, resImg)
        resImg = np.where(imgArray < self.avImg - self.sigmas * self.delta, imgArray, resImg)

        cv2.imshow('delta', self.delta)
        cv2.imshow('average', self.avImg)
        cv2.imshow('average+delta', self.avImg + self.delta)
        cv2.imshow('average-delta', self.avImg - self.delta)

        return resImg
