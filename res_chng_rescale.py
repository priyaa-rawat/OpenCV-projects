import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def change_res(cap,width,height):
    cap.set(3,width)
    cap.set(4,height)

#change_res(1260,480)

def rescale_frame(frame,percent = 75):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent/100)
    height = int(frame.shape[0] * scale_percent/100)
    dim=(width,height)
    return cv2.resize(frame, dim , interpolation = cv2.INTER_AREA)

while True:
    ret , frame = cap.read()
    frame = rescale_frame(frame ,percent=30)
    cv2.imshow('frame',frame)
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = rescale_frame(gray , percent = 70)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()