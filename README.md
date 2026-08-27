# Face Detection using Haar Cascades with OpenCV and Matplotlib
# Name : LIGNESHWAR K
# Reg no : 212223230113

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

## Program :
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load Image
# --------------------------------------------------

img = cv2.imread(r"C:\Users\admin\Downloads\L.jpeg", 0)

if img is None:
    print("Image not found!")
    exit()

# Display original image
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()



roi = img[100:400, 100:400]

plt.imshow(roi, cmap="gray")
plt.title("Extracted ROI")
plt.axis("off")
plt.show()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)



def detect_face(img):

    img_copy = img.copy()

    faces = face_cascade.detectMultiScale(
        img_copy,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            img_copy,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            10
        )

    return img_copy



def detect_eyes(img):

    img_copy = img.copy()

    eyes = eye_cascade.detectMultiScale(
        img_copy,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in eyes:

        cv2.rectangle(
            img_copy,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            10
        )

    return img_copy



face_result = detect_face(img)

plt.imshow(face_result, cmap="gray")
plt.title("Face Detection")
plt.axis("off")
plt.show()


eye_result = detect_eyes(img)

plt.imshow(eye_result, cmap="gray")
plt.title("Eye Detection")
plt.axis("off")
plt.show()



cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to access webcam")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Draw rectangle and label
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    # Display video
    cv2.imshow(
        "Real-Time Face Detection",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
```

## Output :

<img width="342" height="507" alt="image" src="https://github.com/user-attachments/assets/65d9bb33-3efe-4364-8477-884384c71a1e" />
<img width="277" height="510" alt="image" src="https://github.com/user-attachments/assets/52022aa1-3218-4f9c-9569-69b1549e4eba" />
<img width="347" height="507" alt="image" src="https://github.com/user-attachments/assets/14421116-21f5-464e-a9ce-6e51d291f6ab" />
<img width="347" height="502" alt="image" src="https://github.com/user-attachments/assets/b40647f8-f605-4572-95bc-1a24071624e1" />




## Result :

Thus, to write a Python program using OpenCV to perform image manipulations for the given objectives is executed sucessfully.
