import enum
import numpy as np
from eye_utils import maskEye, cropEye
import cv2
from pupil import Pupil


class EyeSide(enum.Enum):
    LEFT_EYE = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE = [42, 43, 44, 45, 46, 47]


class Eye(object):

    def __init__(self, originFrame, landmarks, eyeSide, threshold):
        self.pupil = None
        self.eyeArea = None
        self.eyeAreaCoords = None
        self.verticalCoords = None
        self.horizontalCoords = None
        self.center = None
        self._analyze(originFrame, landmarks, eyeSide, threshold)

    def _analyze(self, originFrame, landmarks, eyeSide, threshold):
        if EyeSide.LEFT_EYE == eyeSide:
            points = EyeSide.LEFT_EYE.value
        elif EyeSide.RIGHT_EYE == eyeSide:
            points = EyeSide.RIGHT_EYE.value
       
        self.horizontalCoords = self.getHorizontalCoords(landmarks, points)
        self.verticalCoords = self.getVerticalCoords(landmarks, points)
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points], np.int32)
        maskedEye = maskEye(originFrame, region)
        self.eyeArea, self.eyeAreaCoords = cropEye(maskedEye, region)

        height, width = self.eyeArea.shape[:2]
        self.center = (width / 2, height / 2)
 
        self.pupil = Pupil(self.eyeArea, threshold) 
            
    def getIrisSpaceRatio(self, irisArea):
        irisArea = irisArea[5:-5, 5:-5]
        height, width = irisArea.shape[:2]
        pixelArea = height * width
        blackArea = pixelArea - cv2.countNonZero(irisArea)
        return blackArea / pixelArea
 
    def _findBestThreshold(self):
        avgIrisSpaceRatio = 0.48
        trials = {}
        for threshold in range(5, 100, 5):
            irisArea = Pupil.imageProcessing(self.eyeArea, threshold)
            trials[threshold] = self.getIrisSpaceRatio(irisArea)
                
        bestThreshold, irisSize = min(trials.items(), key=(lambda p: abs(p[1] - avgIrisSpaceRatio)))
        return bestThreshold

    def hasPupil(self):
        return self.pupil is not None
 
    def midPoint(self, point1, point2):
        return (point1.x + point2.x) // 2, (point1.y + point2.y) // 2
    
    def getHorizontalCoords(self, landmarks, points):
        leftPoint = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        rightPoint = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        return leftPoint, rightPoint
 
    def getVerticalCoords(self, landmarks, points):
        topPoint = self.midPoint(landmarks.part(points[1]), landmarks.part(points[2]))
        bottomPoint = self.midPoint(landmarks.part(points[5]), landmarks.part(points[4]))
        return topPoint, bottomPoint

 



        

 
        

        

        

        
        
        
        
        
        



