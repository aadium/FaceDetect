# FaceDetect
This code performs facial detection on an image using the Haar Cascade classifier and displays the image with the detected faces and the number of faces detected.

First, the user is prompted to enter the name of the image file to be processed. The image is then loaded using OpenCV's imread function.

The image is then converted to grayscale using the cvtColor function with the COLOR_BGR2GRAY flag. This is necessary for the Haar Cascade classifier to detect faces.

The image is then preprocessed using histogram equalization with the equalizeHist function. This enhances the contrast of the image and improves the accuracy of the facial detection.

Next, the Haar Cascade classifier is loaded using the CascadeClassifier function. This classifier is a machine learning-based approach to detect objects in images. In this case, it is used to detect faces.

The detectMultiScale function is then used to detect faces in the preprocessed grayscale image. The scaleFactor parameter specifies how much the image size is reduced at each image scale, while the minNeighbors parameter specifies how many neighbors each candidate rectangle should have to retain it. The function returns a list of detected faces as rectangles.

If faces are detected in the image, a loop is used to draw rectangles around each detected face using the rectangle function. The image with the detected faces and the number of faces detected is then displayed using the imshow and title functions from the Matplotlib library. The cvtColor function is used again to convert the image to RGB format, which is the format required by Matplotlib.

If no faces are detected, a message is printed to the console saying "No faces detected in the image".
