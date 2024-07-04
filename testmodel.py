import cv2 
import time

# Initialize video capture from the default camera
video = cv2.VideoCapture(0)

# Load the face detection model
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the face recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to the recognized IDs, index 0 is reserved for 'unknown'
name_list = ["", "Abhishek"]

# Start the video frame capture loop
while True:
    # Read a frame from the video
    ret, frame = video.read()
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    # Process each face found
    for (x, y, w, h) in faces:
        # Predict the ID of the face
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 50:
            # Face recognized, draw green boxes
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)  # Border
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Box
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)  # Top rectangle
            # Display the name of the recognized person
            cv2.putText(frame, name_list[serial], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Face unknown, draw red boxes
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Border
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Box
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 0, 255), -1)  # Top rectangle
            # Display "Unknown" for unrecognized faces
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display the frame with detected faces
    cv2.imshow("Frame", frame)
    
    # Break the loop if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()