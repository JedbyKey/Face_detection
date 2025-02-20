import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade file was loaded successfully
if face_cascade.empty():
    print("Error: Cascade file not loaded. Please check the path.")
    exit()

# Read the input image
img = cv2.imread('test2.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not loaded. Please check the file path.")
    exit()

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('Detected Faces', img)

# Wait for any key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()