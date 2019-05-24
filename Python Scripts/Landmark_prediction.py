'''
There are 68 landmarks in dlib's shape_predictor function. First predicting landmark then applying it on the camera's frame.
'''
import cv2
import numpy as np
import dlib  #before installing dlib, installation of ctype is required
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cam= cv2.VideoCapture(0)  #Aquiring Webcam

detector= dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _,frame= cam.read()
    frame= cv2.flip(frame,1)
    grayFrame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)     #Converting image to grayscale format
    
    faces= detector(grayFrame)   
    for face in faces:
        
        landmark= predictor(grayFrame,face)   #predicting 68 landmarks 
        for i in range(0,68):              #printing landmarks of lips
            x= landmark.part(i).x
            y= landmark.part(i).y
            cv2.circle(frame,(x,y),2,(0,255,0),-1)
        
    cv2.imshow('Frame', frame)

    key= cv2.waitKey(1)
    if key== 27:
        break
        
cam.release()
cv2.destroyAllWindows()