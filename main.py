import cv2 
from google.colab.patches import cv2_imshow 
# Load the cascade 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
# Read the input image 
img = cv2.imread('/content/Screen Shot 2020-11-05 at 11.06.03 PM.png') 
# Convert into grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# Detect faces 
faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
# Draw rectangle around the faces 
for (x, y, w, h) in faces: 
cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) 
# Display the output 
# cv2.imshow('img', img) 
cv2_imshow(img) 
cv2.waitKey() 
