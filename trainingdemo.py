import cv2
import numpy as np
from PIL import Image
import os

# Create a Local Binary Patterns Histograms recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the dataset containing face images
path = "datasets"

def getImageID(path):
    # Generate a list of image paths in the dataset folder
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []  # List to store face data
    ids = []  # List to store corresponding IDs

    # Loop through all image paths
    for imagePaths in imagePath:
        # Open the image and convert it to grayscale
        faceImage = Image.open(imagePaths).convert('L')
        # Convert the image object to a numpy array
        faceNP = np.array(faceImage, 'uint8')
        # Extract the ID from the filename
        Id = int(os.path.split(imagePaths)[-1].split(".")[1])
        # Append the face numpy array and ID to their respective lists
        faces.append(faceNP)
        ids.append(Id)
        # Display the face being trained on
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)  # Wait for 1 ms before moving on to the next image
    return ids, faces

# Retrieve IDs and face data for training
IDs, facedata = getImageID(path)

# Train the recognizer on the dataset
recognizer.train(facedata, np.array(IDs))

# Save the trained model to a file
recognizer.write("Trainer.yml")

# Destroy all OpenCV windows
cv2.destroyAllWindows()

print("Training Completed............")