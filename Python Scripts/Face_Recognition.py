'''
Predicting the Face in the Webcam, in return we get coordinates of a diagonal of a rectangular region where our face is. 
'''
import cv2
import numpy as np
import dlib  #before installing dlib, installation of ctype is required
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cam= cv2.VideoCapture(0)  #Aquiring Webcam

detector= dlib.get_frontal_face_detector()

while True:
    _,frame= cam.read()
    frame= cv2.flip(frame,1)
    grayFrame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)     #Converting image to grayscale format
    
    faces= detector(grayFrame)        #Detecting face and returning rectangle's diagonal coordinates
    for face in faces:
        x1= face.left()
        y1= face.top()
        x2= face.right()
        y2= face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)    #adding the rectangle to the frame
    
    cv2.imshow('Frame', frame)

    key= cv2.waitKey(1)
    if key== 27:
        break
        
cam.release()
cv2.destroyAllWindows()