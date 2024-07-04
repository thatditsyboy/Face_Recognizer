**Face Recognition System using OpenCV**

This project implements a face recognition system using OpenCV, allowing users to collect face datasets, train a recognition model, and recognize faces in real-time using a webcam.

**Project Overview**

**This project consists of three main components:**

**Dataset Collection:**

The datacollect.py script captures images from a webcam, detects faces using the Haar Cascade classifier, and saves them to a datasets directory for training.

**Model Training:**

The trainer.yml file is generated by trainer.py after training a Local Binary Patterns Histograms (LBPH) recognizer on the collected face images. It associates each face with an ID for future recognition.

**Real-time Face Recognition:**

The recognize.py script loads the trained Trainer.yml model and performs real-time face recognition. Detected faces are outlined with a green box if recognized, or red if unknown.

**Technologies Used:**

OpenCV: Used for capturing video, face detection, and LBPH face recognition.

Python: Programming language for scripting and implementing the project.

PIL (Python Imaging Library): Used for handling image data during dataset collection.

NumPy: Used for numerical operations and array manipulation.
