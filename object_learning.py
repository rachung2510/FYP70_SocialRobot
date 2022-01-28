import sklearn
import argparse
import imutils
import time
import dlib
import cv2
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from jetcam.usb_camera import USBCamera
from emotion_recognition import init_emotion, get_emotion_class
from object_detection import init_obj, get_objects

tic = time.time()
# argparse
parser = argparse.ArgumentParser()
parser.add_argument("dev", help="device no. from 'ls /dev/video*'", type=int)
args = parser.parse_args()

# constants
model_path = 'models/'
emotion_path = model_path + 'EmotionRecognition/'
obj_path = model_path + 'ObjectDetection/'

# Load emotion recognition models
detector, predictor, emotion_models = init_emotion(emotion_path)

# Load object detection models
yolo, yolo_classes = init_obj(obj_path)

# Start camera
print("[INFO] camera sensor warming up...")
width, height = 640, 480
cam = USBCamera(width=width, height=height, capture_device=args.dev)

toc = time.time()
print("Time taken to boot: %.2fs" % (toc-tic))
while True:
    frame = cam.read()
    # if the `q` key was pressed, break from the loop
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#    if not frame.any():
#        print("No frames captured.")
#        break

    frame = get_objects(frame, width, height, yolo, yolo_classes)
    get_emotion_class(frame, detector, predictor, emotion_models)

    # show the output image with the face detections + facial landmarks
#    cv2.imshow("Video", frame)

# do a bit of cleanup
cam.stop()
#cv2.destroyAllWindows()
