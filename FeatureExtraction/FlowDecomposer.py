import cv2
import numpy as np
import matplotlib.pyplot as plt

class Aligner:
    def __init__(self):
        self.detector = cv2.ORB_create(5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def Align(self, keyframeInd, frames):
        keyGray = cv2.cvtColor(frames[keyframeInd], cv2.COLOR_BGR2GRAY)
        kp1, d1 = self.detector.detectAndCompute(keyGray, None)
        newFrames = []
        for i, frame in enumerate(frames):
            if i == keyframeInd:
                newFrames.append(frame)
                continue

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, d2 = self.detector.detectAndCompute(grayFrame, None)

            matches = self.matcher.match(d1, d2)
            matches.sort(key=lambda x: x.distance)
            matches = matches[:int(len(matches)*90)]
            p1 = np.zeros((len(matches), 2))
            p2 = np.zeros((len(matches), 2))

            for i in range(len(matches)):
                p1[i, :] = kp1[matches[i].queryIdx].pt
                p2[i, :] = kp2[matches[i].trainIdx].pt

            homography, mask = cv2.findHomography(p2, p1, cv2.RANSAC)
            transformedFrame = cv2.warpPerspective(frame, homography, frames[keyframeInd].shape[1::-1])
            cv2.imshow("", transformedFrame)
            cv2.imwrite("examples/{}_aligned".format(len(i)), transformedFrame)
            newFrames.append(transformedFrame)

        return newFrames

