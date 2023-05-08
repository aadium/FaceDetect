import cv2
import numpy as np

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video file
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply noise reduction and contrast stretching
    gray = cv2.GaussianBlur(gray, (15,15), 0)
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(20,20))
    gray = clahe.apply(gray)

    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

    # If faces are detected, draw rectangles around them and display the number of faces detected
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            if (w*h)>12000:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    # Display frame with face detection results
    cv2.imshow('Frame', frame)

    # Wait for 1 millisecond for the next frame
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
