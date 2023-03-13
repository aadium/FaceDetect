import cv2
import numpy as np
import matplotlib.pyplot as plt

fileName = input("Please enter the name of the file: ")
type = input("Please enter the file extension: ")

# Load image (LOAD)
img = cv2.imread('images/' + fileName + '.' + type)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess image
gray = cv2.equalizeHist(gray)

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in image (AIM)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 10)

# Display image with detected faces (SHOOT)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
