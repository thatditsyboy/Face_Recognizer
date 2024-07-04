import cv2
import os

# Ensure the datasets directory exists
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

id = input("Enter Your ID: ")
count = 0
max_images = 500

video = cv2.VideoCapture(0)

print("Ensure good lighting and look directly into the camera...")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1

        # Save the captured image
        cv2.imwrite(f'datasets/User.{id}.{count}.jpg', gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        # Display the number of captured images
        cv2.putText(frame, f'Images Captured: {count}/{max_images}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    # Break if maximum images are captured or if 'q' is pressed
    if count >= max_images:
        break
    if k == ord('q'):
        print("Interruption received, stopping...")
        break

video.release()
cv2.destroyAllWindows()
print(f"Dataset Collection Done. {count} images saved.")
