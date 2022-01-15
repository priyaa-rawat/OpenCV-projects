import cv2
import numpy as np

#to capture the camera
cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('7.jpg')
#capturing the video saved
myVid = cv2.VideoCapture('bhuvan.mp4')

detection = False #tells if we have targetimg or not
frameCounter = 0 #no. of frames we have displayed from our video

sucess, imgVideo = myVid.read()
#getting the height of the image
hT, wT, cT = imgTarget.shape
#resizing the video according to the image
imgVideo = cv2.resize(imgVideo, (wT, hT))

#detector
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)


#for webcam
while True:
    sucess, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam,kp2,None)

    if detection is False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    elif frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    sucess, imgVideo = myVid.read()
    imgVideo = cv2.resize(imgVideo, (wT, hT))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2) #shows the matching features btw target and imgwebcam

    #homography
    if len(good) > 20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)#Ransac algorithm , threshold value = 5
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix) #points where the model assumes the image is found
        img2 = cv2.polylines(imgWebcam,np.int32(dst), True, (255, 0 ,255), 3)#img that shows the boundary of the targetimg

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))#video according to img2

        maskNew = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))#masking the area where img is found in webcam img
        maskInv = cv2.bitwise_not(maskNew)#inversing the mask to cover area where no img was found
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv) #filling colors in non-img area according to real bg in video
        imgAug = cv2.bitwise_or(imgWarp, imgAug) #filling augmented image in warp



    cv2.imshow('maskNew', imgAug)
    cv2.waitKey(1) 
    frameCounter += 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


