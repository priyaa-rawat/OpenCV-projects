import numpy as np
import cv2
from PIL import Image
import os

current_id=0
label_ids = {}
y_labels = []
x_train = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")
face_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_frontalface_alt2.xml')

for root , dirs , files in image_dir:
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root,file)
            label= os.path.basename(root).replace('.','-').lower()
            print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            print(label_ids)
        
            #y_labels.append(label) #some number
            #x_train.append(path) #verify this image , turn into anumpy array , GRAY
            pil_image = Image.open(path).convet('L') #grayscale
            image_array = np.array(pil_image,"uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor = 1.5 , minNeighbours =5)

            for (x,y,w,h) in image_array:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)



  


cv2.destroyAllWindows()