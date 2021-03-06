import os
import numpy as np
import cv2

filename = 'video.mp4'  # '.avi' or '.mp4'
frames_per_seconds = 30.0
my_res = '360p'

def change_res(cap,width,height):
    cap.set(3,width)
    cap.set(4,height)

STD_Dimensions = {
    '480p': (640,480),
    '720p': (1280,720),
    '1080p': (1920,1080),
    '4k': (3840,2160) ,
}

def get_dims (cap,res='1080p'):
    width,height = STD_Dimensions['480p']
    if res in STD_Dimensions:
        width,height = STD_Dimensions[res]
    change_res(cap,width,height)
    return width,height

VIDEO_TYPE = {
    'avi' : cv2.VideoWriter_fourcc(*'XVID'),
    #'.mp4' : cv2.VideoWriter_fourcc(*'H264'),
    'mp4' : cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename , ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

cap = cv2.VideoCapture(0)
dims = get_dims(cap , res = my_res)
VIDEO_TYPE_CV2 = get_video_type(filename)
out = cv2.VideoWriter(filename , VIDEO_TYPE_CV2 ,frames_per_seconds,dims)

while True:
    ret , frame = cap.read()

    out.write(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q') :
        break

cap.release()
out.release()
cv2.destroyAllWindows()