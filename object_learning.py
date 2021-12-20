import sys
emotion_path = 'EmotionRecognition/'
obj_path = 'ObjectDetection/'
sys.path.append(emotion_path)
import realtime_emotion_recognition as emotion_recognition
sys.path.append('../'+obj_path)
import video as object_detection
sys.path.pop()

from imutils import face_utils
import imutils
import time
import dlib
import cv2
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Load emotion models and predictors
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use tensorflow cpu (doesn't work on gpu)

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(emotion_path+'shape_predictor_68_face_landmarks.dat')

# load emotion models
model_path = emotion_path + 'trained_models/'
print("[INFO] loading emotion recognition models...")
cnn3 = load_model(model_path+'emotion-cnn3.hd5')
svm2 = pickle.load(open(model_path+'emotion-svm2', 'rb'))
cnnA = load_model(model_path+'emotion-cnnA.hd5')
emotion_models = [cnn3, svm2, cnnA]

# Load object detection models
print("[INFO] loading object detection model...")
yolo = cv2.dnn.readNet(obj_path+'./yolov4.weights', obj_path+'./yolov4.cfg')
yolo_classes = []
with open(obj_path+"./coco.names","r") as f:
    yolo_classes = f.read().splitlines()
print((yolo_classes))

# Start camera
print("[INFO] camera sensor warming up...")
cam = cv2.VideoCapture(0)
if (cam.isOpened() == False):
    print("Unable to read camera feed.")
width = int(cam.get(3))
height = int(cam.get(4))

while True:
    ret,frame = cam.read()
    if ret == True:
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not frame.any():
        print("No frames captured.")
        break

    frame = object_detection.get_objects(frame, width, height, yolo, yolo_classes)
    frame = emotion_recognition.get_emotion(frame, detector, predictor, emotion_models)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Video", frame)

# do a bit of cleanup
cam.release()
cv2.destroyAllWindows()
