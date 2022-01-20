from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from helper_functions import mag, angle, getEmotionClass

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use tensorflow cpu (doesn't work on gpu)

# define constants
model_path = 'trained_models/'
emotion_classes = ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
dim = 50

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load models
print("[INFO] loading models...")
##cnn1 = load_model(model_path+'emotion-cnn1.hd5')
cnn2 = load_model(model_path+'emotion-cnn2.hd5')
##cnn3 = load_model(model_path+'emotion-cnn3.hd5')
##cnn4 = load_model(model_path+'emotion-cnn4.hd5')
##cnn5 = load_model(model_path+'emotion-cnn5.hd5')
##cnn6 = load_model(model_path+'emotion-cnn6.hd5')

##svm1 = pickle.load(open(model_path+'emotion-svm1', 'rb'))
svm2 = pickle.load(open(model_path+'emotion-svm2', 'rb'))
##svm3 = pickle.load(open(model_path+'emotion-svm3', 'rb'))
##svm4 = pickle.load(open(model_path+'emotion-svm4', 'rb'))

cnnA = load_model(model_path+'emotion-cnnA.hd5')
cnnB = load_model(model_path+'emotion-cnnB.hd5')


def start():
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        vectors, coords = [], []
        cnn_v_input = np.array([])
        svm_vc_input = np.array([])
        cnn_px_input = np.array([])

        frame = vs.read()   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract features
            cog = tuple(shape.mean(axis=0).astype(int))
            for (x,y) in shape:
                cv2.line(frame, (x,y), cog, (255,0,0), 1) # draw vector lines
                cv2.circle(frame, (x,y), 1, (0,0,255), -1) # draw markers
                vectors.append([mag(cog, (x,y)), angle(cog, (x,y))]) # get vector magnitude and direction
                coords.append([x-cog[0], y-cog[1]])
            cv2.circle(frame, cog, 5, (255,255,0), -1) # draw center of gravity

            # get bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # show the face number

            # model inputs
            vectors = np.array(vectors)
            scale_factor = 1 / max(vectors[:,0])
            vectors[:,0] = vectors[:,0] * scale_factor # normalize magnitudes
            vectors = vectors[:,0] * vectors[:,1]
            coords = (np.array(coords) * scale_factor).reshape(-1) # 1D array

            ## vectors only            
            cnn_v_input = vectors.reshape(1, len(vectors), 1)
            svm_v_input = vectors.reshape(1,-1)

            ## vectors & coords            
            cnn_vc_input = np.r_[vectors, coords].reshape(1, len(vectors)+len(coords), 1)
            svm_vc_input = np.r_[vectors, coords].reshape(1,-1)
            
            ## image graylevels
            cnn_px_input = frame[y:y+h, x:x+w] # crop to face
            cnn_px_input = cv2.cvtColor(cnn_px_input, cv2.COLOR_RGB2GRAY) # convert to grayscale
            cnn_px_input = cv2.equalizeHist(cnn_px_input) # equalize histogram
            cnn_px_input = imutils.resize(cnn_px_input, width=int(dim*1.05)) # buffer of 5 pixels for cropping to 100x100
            cnn_px_input = np.expand_dims(cnn_px_input[:dim,:dim], axis=0) # shape=(1,dim,dim)
##            cnn_px_input = np.expand_dims(cnn_px_input, axis=-1) # if min ndim=4
            
            # prediction
            if cnn_v_input.size!=0 and svm_vc_input.size!=0 and cnn_px_input.size!=0:
                cnn2_prob = cnn2.predict(cnn_v_input)
                svm2_prob = 2 * svm2.predict_proba(svm_vc_input)
                cnnA_prob = 0.5 * cnnA.predict(cnn_px_input)
                cnnB_prob = 0.5 * cnnB.predict(cnn_px_input)
                emotion_class = emotion_classes[getEmotionClass(np.r_[cnn2_prob, svm2_prob, cnnB_prob, cnnB_prob], \
                                                                emotion_classes)]
                
                cv2.putText(frame, emotion_class.upper(), (x-20,y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)        
                    
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

start()    
