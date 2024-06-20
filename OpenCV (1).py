#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


cv2.__version__


# In[3]:


img = cv2.imread('C:\\Users\\v.omsai\\opencv\\Modi.jpg',1)
img


# In[4]:


cv2.imshow('PM',img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[5]:


img = cv2.imread('C:\\Users\\v.omsai\\opencv\\Modi.jpg')
resized_img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('modi',gray)
cv2.waitKey()
cv2.destroyAllWindows()


# In[6]:


face_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_frontalface_default.xml')
image = cv2.imread('C:\\Users\\v.omsai\\opencv\\Modi.jpg')
image = cv2.resize(image,(500,500))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray,1.05,5)


# In[7]:


print(faces)


# In[8]:


face_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_frontalface_default.xml')
image = cv2.imread('C:\\Users\\v.omsai\\opencv\\Modi.jpg')
image = cv2.resize(image,(500,500))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.05,5)

if faces is ():
    print('No Faces Found')
    
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,100),1)
    
cv2.imshow('Face Detection',image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[9]:


face_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_eye.xml')
img = cv2.imread('C:\\Users\\v.omsai\\opencv\\Modi.jpg')
resized_img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.3,5)

if faces is ():
    print('No Face Found')

for (x,y,w,h) in faces:
    cv2.rectangle(resized_img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = resized_img[y:y+h,x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv2.imshow('img',resized_img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[10]:


import cv2
video = cv2.VideoCapture(0)

while True:
    check,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('video',gray)
    if cv2.waitKey(1) ==ord('q'):
        break
video.release()
cv2.destroyAllWindows()


# In[11]:


face_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_eye.xml')

def detect(gray,frame):
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame
            
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('video',canvas)
    if cv2.waitKey(1) == ord('q'):
      break

video.release()
cv2.destroyAllWindows()


# In[12]:


body_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_fullbody.xml')
cap = cv2.VideoCapture('C:\\Users\\v.omsai\\Downloads\\walking.avi')

while cap.isOpened():
    check,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('Pedestrains',frame)
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[13]:


import time
import cv2

car_classifier = cv2.CascadeClassifier('C:\\Users\\v.omsai\\Downloads\\haarcascade_car.xml')
cap = cv2.VideoCapture('C:\\Users\\v.omsai\\Downloads\\cars.avi')

while cap.isOpened():
    
    time.sleep(.05)
    
    check,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cars = car_classifier.detectMultiScale(gray,1.3,2)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('cars',frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




