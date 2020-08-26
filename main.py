import cv2
from gaze_tracking import GazeTracking
import pandas as pd

def plotPupil(x, y):
    circleColor = (0, 168, 255)
    lineColor = (0, 255, 0)
    cv2.circle(img, (x, y), 4, circleColor)
    cv2.line(img, (x - 5, y), (x + 5, y), lineColor)
    cv2.line(img, (x, y - 5), (x, y + 5), lineColor)

def plotHVEye(topPoint, bottomPoint, leftPoint, rightPoint):
    cv2.line(img, leftPoint, rightPoint, (0, 255, 255), 1)
    cv2.line(img, topPoint, bottomPoint, (0, 255, 255), 1)  

def plotBox(topPoint, bottomPoint, leftPoint, rightPoint):
    cv2.rectangle(img, (leftPoint[0], topPoint[1]), (rightPoint[0], bottomPoint[1]), (0, 255, 255), 1) 

if __name__ == '__main__':
    gaze = GazeTracking()
    cap = cv2.VideoCapture(0)
    l_x = []
    l_y = []
    r_x = []
    r_y = []
    while cap.isOpened():
        _, frame = cap.read()
        gaze.analyze(frame)
        
        img = frame.copy()
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if gaze.isPupilsLocated():
            xl, yl = gaze.getLeftPupilCoords()
            #l_x.append(xl)
            #l_y.append(yl)
            topPoint, bottomPoint = gaze.leftEye.verticalCoords
            leftPoint, rightPoint = gaze.leftEye.horizontalCoords
            plotHVEye(topPoint, bottomPoint, leftPoint, rightPoint)
            plotPupil(xl, yl)
            #plotBox(topPoint, bottomPoint, leftPoint, rightPoint)

            xr, yr = gaze.getRightPupilCoords()
            #r_x.append(xr)
            #r_y.append(yr)
            topPoint, bottomPoint = gaze.rightEye.verticalCoords
            leftPoint, rightPoint = gaze.rightEye.horizontalCoords
            plotHVEye(topPoint, bottomPoint, leftPoint, rightPoint)
            plotPupil(xr, yr)
            plotBox(topPoint, bottomPoint, leftPoint, rightPoint)
        
            if xl in range(255, 279) and yl in range(224, 249) and xr in range(368, 392) and yr in range(217, 240):
                print('in')
            else:
                print('out')
        cv2.imshow('frame', img)
        if cv2.waitKey(1) == 27:
            break

    d = { 
          'LeftX': l_x,
          'LeftY': l_y,
          'RightX': r_x,
          'RightY': r_y 
    }
  
    df = pd.DataFrame(data=d)
    print(df.head())
    #df.to_csv('out.csv')
