import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eyeArea, threshold):
        self.irisArea = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detectIris(eyeArea)

    @staticmethod
    def imageProcessing(eyeArea, threshold):
        """Performs operations on the eye frame to isolate the iris
        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame
        Returns:
            A frame with a single element representing the iris
        """ 
        newFrame = cv2.threshold(eyeArea, threshold, 255, cv2.THRESH_BINARY)[1]
        newFrame = cv2.erode(newFrame, None, iterations=2)
        newFrame = cv2.dilate(newFrame, None, iterations=4)
        newFrame = cv2.medianBlur(newFrame, 5)
        return newFrame

    def detectIris(self, eyeArea):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.
        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.irisArea = self.imageProcessing(eyeArea, self.threshold)

        contours, _ = cv2.findContours(self.irisArea, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass

   
