import sklearn # needed to avoid ImportError: cannot allocate memory in static TLS block
import argparse
from imutils import face_utils
import imutils
import time
import dlib
import cv2
from tensorflow.keras.models import load_model
import pickle
from jetcam.usb_camera import USBCamera
import numpy as np
import math

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("dev", help="device no. from 'ls /dev/video*'", type=int)
args = parser.parse_args()

# define constants
model_path = 'models/'
emotion_classes = ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
dim = 50

def init_emotion():
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path + 'shape_predictor_68_face_landmarks.dat')

    print("[INFO] loading models...")
    cnn2 = load_model(model_path + 'emotion-cnn2.hd5')
    svm2 = pickle.load(open(model_path + 'emotion-svm2', 'rb'))
    cnnA = load_model(model_path + 'emotion-cnnA.hd5')
    cnnB = load_model(model_path + 'emotion-cnnB.hd5')

    print("[INFO] camera sensor warming up...")
    vs = USBCamera(capture_device=args.dev)

    return vs, detector, predictor, (cnn2,svm2,cnnA,cnnB)

def get_emotion_class(frame, detector, predictor, models):
    (cnn2,svm2,cnnA,cnnB) = models
    emotion_class = "neutral"

    cnn_v_input = np.array([])
    svm_vc_input = np.array([])
    cnn_px_input = np.array([])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    face_area = 0
    for (i, rect) in enumerate(rects):
        vectors, coords = [], []
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract features
        cog = tuple(shape.mean(axis=0).astype(int))
        for (x,y) in shape:
            vectors.append([mag(cog, (x,y)), angle(cog, (x,y))])
            coords.append([x-cog[0], y-cog[1]])

        # get bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # model inputs
        vectors = np.array(vectors)
        scale_factor = 1 / max(vectors[:,0])
        vectors[:,0] = vectors[:,0] * scale_factor
        vectors = vectors[:,0] * vectors[:,1]
        coords = (np.array(coords) * scale_factor).reshape(-1) # 1D array

        # actions for main face (largest area)
        if w*h >= face_area:
            face_area = w*h # store new max face_area

            ## vectors
            cnn_v_input = vectors.reshape(1, len(vectors), 1)
            svm_vc_input = np.r_[vectors, coords].reshape(1,-1)

            ## image graylevels
            cnn_px_input = frame[y:y+h, x:x+w] # crop to face
            cnn_px_input = cv2.cvtColor(cnn_px_input, cv2.COLOR_RGB2GRAY) # convert to grayscale
            cnn_px_input = cv2.equalizeHist(cnn_px_input) # equalize histogram
            cnn_px_input = imutils.resize(cnn_px_input, width=int(dim*1.1)) # buffer for cropping
            cnn_px_input = np.expand_dims(cnn_px_input[:dim,:dim], axis=0) # shape=(1,dim,dim)
            cnn_px_input = np.expand_dims(cnn_px_input, axis=-1) # for some reason, tts_link_test tensorflow needs another dim

    # prediction
    if cnn_v_input.size!=0 and svm_vc_input.size!=0 and cnn_px_input.size!=0:
        cnn2_prob = cnn2.predict(cnn_v_input)
        svm2_prob = 2 * svm2.predict_proba(svm_vc_input)
        if cnn_px_input.shape[1] == 50 and cnn_px_input.shape[2] == 50:
            cnnA_prob = 0.5 * cnnA.predict(cnn_px_input)
            cnnB_prob = 0.5 * cnnB.predict(cnn_px_input)
            probArr = np.r_[cnn2_prob, svm2_prob, cnnA_prob, cnnB_prob]
        else:
            probArr = np.r_[cnn2_prob, svm2_prob]
        prob = np.sum(probArr, axis=0)
        emotion_class = emotion_classes[np.argmax(prob)]

    return emotion_class

def mag(pointA, pointB):
    x = pointA[0] - pointB[0]
    y = pointA[1] - pointB[1]
    return math.sqrt(x*x + y*y)

def angle(cog, point):
    x = point[0] - cog[0]
    y = point[1] - cog[1]

    if not x:
        return math.pi/2 if y>0 else -math.pi/2

    angle = math.atan(y/x)
    if x<0 and y>0: # 2nd quadrant
        angle += math.pi
    elif x<0 and y<0: # 3rd quadrant
        angle -= math.pi
    return angle

vs, detector, predictor, models = init_emotion()
while True:
    frame = vs.read()
    get_emotion_class(frame, detector, predictor, models)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#cv2.destroyAllWindows()
vs.stop()
