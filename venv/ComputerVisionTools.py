import numpy as np
import cv2

class Tools:
    def __init__(self):
        self.version = 0.1

        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))



    def imageShower(self):
        # Our operations on the frame come here
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('image', gray)




    def edgeDetector_opencv(self):
        edges = cv2.Canny(self.frame, 100, 200)

        # Display the resulting frame
        cv2.imshow('edgeDet', edges)




    def opticalFlow_opencv(self):
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.lastFrame)

        lastFrame_grey = cv2.cvtColor(self.lastFrame, cv2.COLOR_BGR2GRAY)
        thisFrame_grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(lastFrame_grey, mask=None, **self.feature_params)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(lastFrame_grey, thisFrame_grey, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
            frame = cv2.circle(self.frame, (a, b), 5, self.color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        cv2.imshow('opticalFlow', img)


    def denseOpticalFlow_ocv(self):

        lastFrame_grey = cv2.cvtColor(self.lastFrame, cv2.COLOR_BGR2GRAY)
        thisFrame_grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(lastFrame_grey, thisFrame_grey, None, 0.5, 3, 5, 3, 2, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('denseOF', rgb)



    def start(self):

        cap = cv2.VideoCapture(0)
        self.ret, self.frame = cap.read()

        self.hsv = np.zeros_like(self.frame)
        self.hsv[..., 1] = 255

        while (True):
            # Set the last captured frame to the last frame
            self.lastFrame = self.frame

            # Capture frame-by-frame
            self.ret, self.frame = cap.read()

            #self.imageShower()             # Display the base image
            #self.edgeDetector_opencv()     # Display the detected edge
            #self.opticalFlow_opencv()      # Display optical flow
            self.denseOpticalFlow_ocv()    # Display dense optical flow

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()