import cv2
import numpy as np
import matplotlib.pyplot as plt

fileName = input("Please enter the name of the file with the extension: ")

# Load image (LOAD)
img = cv2.imread('images/' + fileName)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply noise reduction and contrast stretching
gray = cv2.GaussianBlur(gray, (5,5), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in image (AIM)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# If faces are detected, draw rectangles around them and display the number of faces detected (SHOOT)
if len(faces) > 0:
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 10)

    # Display image with detected faces and number of faces detected
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(str(len(faces)) + ' face(s) detected')
    plt.show()
    
# If no faces are detected, display a message
else:
    print("No faces detected in the image.")
