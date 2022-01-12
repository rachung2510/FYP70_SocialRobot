from imutils.video import VideoStream
from imutils import face_utils
##import argparse
import imutils
import time
import dlib
import cv2
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from helper_functions import mag, angle, getEmotionClass

import os
##os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use tensorflow cpu (doesn't work on gpu)

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

## best performing: cnn2, svm2, cnnB

def start():
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    # loop over the frames from the video stream
    while True:
        # store vectors as input data for model prediction
        vectors, coords = [], []
        classes = {} 
        
        # grab frame, and convert to grayscale
        frame = vs.read()   
##        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # get center of gravity (COG)
            cog = tuple(shape.mean(axis=0).astype(int))
            
            # draw (x, y)-coordinates for the facial landmarks
            for (x,y) in shape:
                cv2.line(frame, (x,y), cog, (255,0,0), 1) # draw vector lines
                cv2.circle(frame, (x,y), 1, (0,0,255), -1) # draw markers
                vectors.append([mag(cog, (x,y)), angle(cog, (x,y))]) # get vector magnitude and direction
                coords.append([x-cog[0], y-cog[1]])            
            cv2.circle(frame, cog, 5, (255,255,0), -1) # draw center of gravity

            # convert dlib's rectangle to a OpenCV-style bounding box (x,y,w,h)
            # and draw face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)            
            cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # show the face number

            # model inputs
            ## vectors & coords
            vectors = np.array(vectors)
            scale_factor = 1 / max(vectors[:,0])
            vectors[:,0] = vectors[:,0] * scale_factor # normalize magnitudes
            vectors = vectors[:,0] * vectors[:,1]
            coords = (np.array(coords) * scale_factor).reshape(-1) # 1D array

            # vectors only
            svm1_input = vectors.reshape(1,-1)
            cnn1_input = vectors.reshape(1, len(vectors), 1)

            # vectors & coords
            svm2_input = np.r_[vectors, coords].reshape(1,-1)
            cnn3_input = np.r_[vectors, coords].reshape(1, len(vectors)+len(coords), 1)
    #         print(svm_input.shape, cnn_input.shape)
            
            ## image graylevels
            cnnA_input = frame[y:y+h, x:x+w] # crop to face
            cnnA_input = cv2.cvtColor(cnnA_input, cv2.COLOR_RGB2GRAY) # convert to grayscale
            cnnA_input = cv2.equalizeHist(cnnA_input) # equalize histogram
            cnnA_input = imutils.resize(cnnA_input, width=int(dim*1.05)) # buffer of 5 pixels for cropping to 100x100
            cnnA_input = np.expand_dims(cnnA_input[:dim,:dim], axis=0) # shape=(1,dim,dim)
            
            # prediction
##            classes['CNN1'] = emotion_classes[np.argmax(cnn1.predict(cnn1_input))]
            classes['CNN2'] = emotion_classes[np.argmax(cnn2.predict(cnn1_input))]
##            classes['CNN3'] = emotion_classes[np.argmax(cnn3.predict(cnn3_input))]
##            classes['CNN4'] = emotion_classes[np.argmax(cnn4.predict(cnn1_input))]
##            classes['CNN5'] = emotion_classes[np.argmax(cnn5.predict(cnn1_input))]
##            classes['CNN6'] = emotion_classes[np.argmax(cnn6.predict(cnn3_input))]
                        
##            classes['SVM1'] = emotion_classes[svm1.predict(svm1_input)[0]]
            classes['SVM2'] = emotion_classes[svm2.predict(svm2_input)[0]]
##            classes['SVM3'] = emotion_classes[svm3.predict(svm1_input)[0]]
##            classes['SVM4'] = emotion_classes[svm4.predict(svm2_input)[0]]

            classes['CNNA'] = emotion_classes[np.argmax(cnnA.predict(cnnA_input))]
            classes['CNNB'] = emotion_classes[np.argmax(cnnB.predict(cnnA_input))]

            cnn2_prob = cnn2.predict(cnn1_input)
            svm2_prob = 2 * svm2.predict_proba(svm2_input)
            cnnA_prob = 0.5 * cnnA.predict(cnnA_input)
            cnnB_prob = 0.5 * cnnB.predict(cnnA_input)
            emotion_class = emotion_classes[getEmotionClass(np.r_[cnn2_prob, svm2_prob, cnnA_prob, cnnB_prob], \
                                                            emotion_classes)]
            classes['FINAL'] = emotion_class
            
##            i = 1
##            for k,v in classes.items():
##                cv2.putText(frame, '%s: %s' % (k,v), (x-20,y+h+20*i),
##                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
##                i += 1
            cv2.putText(frame, emotion_class.upper(), (x-20,y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
                    
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

start()    
