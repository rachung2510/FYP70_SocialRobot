from imutils.video import VideoStream
from imutils import face_utils
import imutils, time, dlib, cv2
import torch
import numpy as np
from math import pi
import os, sys, pickle

sys.path.append(os.path.abspath('../'))
from helper_functions import mag, angle
from helper_classes import FCNNModel

# define constants
model_path = '../models/'
emotion_classes = ['anger','disgust','fear','happiness','sadness','surprise','neutral']

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

# load models
print("[INFO] loading models...")
svm_ck = pickle.load(open(model_path+'svm_ck', 'rb'))
fcnn_ck = FCNNModel(68*4, 128, 7) # input, hidden, output
fcnn_ck.load_state_dict(torch.load(model_path + 'FCNN_norm_128_fer.pt', map_location='cpu')['state_dict'])
fcnn_ck.eval()

def start():
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    c, d = 0, 0
    label_ck, label_fer = '', ''
    while True:
        vectors, coords = [], []
        pred = 6

        image = vs.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detector(image, 0)

        if not len(rects):
            cv2.imshow("Frame", image)
            continue

        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)        

        cog = tuple(shape.mean(axis=0).astype(int)) # get center of gravity (COG)            
        for (x,y) in shape:
##            cv2.line(image, (x,y), cog, (0,0,255), 1) # draw vector lines
            cv2.circle(image, (x,y), 2, (255,0,0), -1) # image, center-coords, radius, colour, thickness(fill)
            cv2.circle(image, cog, 5, (0,255,255), -1)    
            vectors.append([mag(cog, (x,y)), angle(cog, (x,y))]) # get vector magnitude and direction                
            coords.append([x,y]) # append coordinates relative to cog            
            
        (x,y,w,h) = face_utils.rect_to_bb(rect) # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x,y,w,h)]
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) # draw the face bounding box

        # input
        vectors = np.array(vectors)
        vectors[:,0] /= max(vectors[:,0]) # normalize magnitudes
        vectors[:,1] = (vectors[:,1] + pi) / (2*pi) # normalize direction
        vectors = vectors.reshape(-1)
        coords = np.array(coords)
        coords -= np.c_[min(coords[:,0]), min(coords[:,1])]
        coords = coords / np.c_[max(coords[:,0]), max(coords[:,1])]
        coords = coords.reshape(-1)
        Vector = np.dstack((vectors, coords)).reshape(1, -1) # shape=(samples, vectors+coords)
                
        # prediction
        pred_svm = svm_ck.predict(Vector)[0]
        with torch.no_grad():
            pred_tensor_fcnn = fcnn_ck(torch.Tensor(Vector))
            pred_fcnn = pred_tensor_fcnn.argmax().numpy().item()

        if pred_svm==4:
            if pred_tensor_fcnn[0][4]>=1.45 or pred_tensor_fcnn[0][6]<=1.9:
                pred = 4            
        elif pred_svm == pred_fcnn: # happiness, surprise, neutral
            pred = pred_svm
        elif pred_svm<3 or pred_fcnn==0:            
            pred = pred_svm if pred_fcnn==6 else pred_fcnn

        # stabilisation
        if not c:
            pred_label_ck = emotion_classes[pred]
            c += 1
        elif c == 3:
            label_ck = pred_label_ck
            c = 0
        else:
            pred_label_ck = emotion_classes[pred] if (pred_label_ck == emotion_classes[pred]) else ''            
            c = c+1 if pred_label_ck else 0
        cv2.putText(image, label_ck, (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

##        if not d:
##            pred_label_fer = emotion_classes[pred_fcnn]
##            d += 1
##        elif d == 3:
##            label_fer = pred_label_fer
##            d = 0
##        else:
##            pred_label_fer = emotion_classes[pred_fcnn] if (pred_label_fer == emotion_classes[pred_fcnn]) else ''            
##            d = d+1 if pred_label_fer else 0
##        cv2.putText(image, 'FCNN: %s'%label_fer, (x,y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
        # show frame            
        cv2.imshow("Frame", image)

        # break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

start()    
