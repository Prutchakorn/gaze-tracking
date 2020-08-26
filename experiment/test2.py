import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)

MODEL_PATH = '../shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]


def image_processing(eye_frame, threshold):
    """Performs operations on the eye frame to isolate the iris
    Arguments:
    eye_frame (numpy.ndarray): Frame containing an eye and nothing else
    threshold (int): Threshold value used to binarize the eye frame
    Returns:
    A frame with a single element representing the iris
    """
    kernel = np.ones((3, 3), np.uint8)
    new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
    new_frame = cv2.erode(new_frame, kernel, iterations=3)
    new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

    return new_frame


def find_best_threshold(eye_frame):
    global iris_size
    """Calculates the optimal threshold to binarize the
    frame for the given eye.
    Argument:
    eye_frame (numpy.ndarray): Frame of the eye to be analyzed
    """
    average_iris_size = 0.48
    trials = {}

    for threshold in range(5, 100, 5):
        iris_frame = image_processing(eye_frame, threshold)
        frame = iris_frame
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        
       
        trials[threshold] = nb_blacks / nb_pixels
        #trials[threshold] = float(iris_size(iris_frame))
        
    best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
    return best_threshold


def iris_size(frame):
    """Returns the percentage of space that the iris takes up on
    the surface of the eye.
    Argument:
        frame (numpy.ndarray): Binarized iris frame
    """
    frame = frame[5:-5, 5:-5]
    height, width = frame.shape[:2]
    nb_pixels = height * width
    nb_blacks = nb_pixels - cv2.countNonZero(frame)
    
    return nb_blacks / nb_pixels

def detect_iris(eye_frame, threshold):
    """Detects the iris and estimates the position of the iris by
    calculating the centroid.
    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
    """
    iris_frame = image_processing(eye_frame, threshold)

    contours, _ = cv2.findContours(iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = sorted(contours, key=cv2.contourArea)

    try:
        moments = cv2.moments(contours[-2])
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        return x, y
    except (IndexError, ZeroDivisionError):
        pass


while cap.isOpened():
    a, frame = cap.read()
    img = frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # get only one face
    faces = detector(img)


    if len(faces) > 0:
        landmarks = predictor(img, faces[0])   
        points = LEFT_EYE_POINTS 
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points], np.int32)
        
        # mask to get only the eye
        height, width = img.shape[:2]
        black_img = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_img, img.copy(), mask=mask)
       
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin
       
        kernel = np.ones((3, 3), np.uint8)
        eye_frame = eye[min_y:max_y, min_x:max_x]
        origin = (min_x, min_y)
        threshold = find_best_threshold(eye_frame)
        print(threshold)
            
        pupil_x, pupil_y = detect_iris(eye_frame, threshold)

        if pupil_x is not None and pupil_y is not None:
            x = origin[0] + pupil_x
            y = origin[1] + pupil_y
            cv2.circle(eye_frame, (x, y), 5, (0, 255,0))
        cv2.imshow('eye', eye_frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
