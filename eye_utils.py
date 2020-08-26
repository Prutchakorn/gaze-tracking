import numpy as np
import cv2


def maskEye(frame, region):
    height, width = frame.shape[:2]
    black_frame = np.zeros((height, width), np.uint8)
    mask = np.full((height, width), 255, np.uint8)
    cv2.fillPoly(mask, [region], (0, 0, 0))
    return cv2.bitwise_not(black_frame, frame.copy(), mask=mask)


def cropEye(eye, region):
    margin = 5
    min_x = np.min(region[:, 0]) - margin
    max_x = np.max(region[:, 0]) + margin
    min_y = np.min(region[:, 1]) - margin
    max_y = np.max(region[:, 1]) + margin
    return eye[min_y:max_y, min_x:max_x], (min_x, min_y)
