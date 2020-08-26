import cv2
from eye import Eye, EyeSide 
import dlib


class GazeTracking(object):

    def __init__(self):
        self._faceDetector = dlib.get_frontal_face_detector()

        MODEL_PATH = './shape_predictor_68_face_landmarks.dat'
        self._predictor = dlib.shape_predictor(MODEL_PATH)
        self.face = None
        self.leftEye = None
        self.rightEye = None
        self.leftEyeThresholds = []
        self.rightEyeThresholds = []
        self.leftEyeThreshold = 0
        self.rightEyeThreshold = 0
    
    def analyze(self, frame):
        newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._faceDetector(newFrame)

        if self._hasFaces(faces):
            self.face = faces[0]
            landmarks = self._predictor(newFrame, self.face)
            self.leftEye = Eye(newFrame, landmarks, EyeSide.LEFT_EYE, self.leftEyeThreshold)
            self.rightEye = Eye(newFrame, landmarks, EyeSide.RIGHT_EYE, self.rightEyeThreshold)

            if not self.isEnoughThresholds():
                self.leftEyeThresholds.append(self.leftEye._findBestThreshold())
                
                self.rightEyeThresholds.append(self.rightEye._findBestThreshold())
            self.rightEyeThreshold = int(sum(self.rightEyeThresholds) / len(self.rightEyeThresholds))
            self.leftEyeThreshold = int(sum(self.leftEyeThresholds) / len(self.leftEyeThresholds))
        else:
            self.face = None 
            self.leftEye = None
            self.rightEye = None

    def _hasFaces(self, faces):
        return len(faces) > 0

    def hasEyes(self):
        return self.leftEye is not None and self.rightEye is not None

    def isPupilsLocated(self):
        if self.hasEyes():
            if self.leftEye.hasPupil() and self.rightEye.hasPupil():
                return self.leftEye.pupil.x is not None and self.rightEye.pupil.y is not None
        return False
   
    def getLeftPupilCoords(self): 
        if self.isPupilsLocated():
            x = self.leftEye.eyeAreaCoords[0] + self.leftEye.pupil.x
            y = self.leftEye.eyeAreaCoords[1] + self.leftEye.pupil.y
            return (x, y)
        return None

    def getRightPupilCoords(self):
        if self.isPupilsLocated():
            x = self.rightEye.eyeAreaCoords[0] + self.rightEye.pupil.x
            y = self.rightEye.eyeAreaCoords[1] + self.rightEye.pupil.y
            return (x, y)
        return None

    def isEnoughThresholds(self):
        return len(self.leftEyeThresholds) >= 30 and len(self.rightEyeThresholds) >= 30
