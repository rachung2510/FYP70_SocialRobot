from imutils.video import VideoStream
from imutils import face_utils
import imutils, time, dlib, cv2
import torch
import numpy as np
import os, sys

sys.path.append(os.path.abspath('../'))
from helper_functions import resize, to_device, get_default_device
from helper_classes import ResNet

# define constants
model_path = '../models/'
emotion_classes = ['anger','disgust','fear','happiness','sadness','surprise','neutral']

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

# load models
print("[INFO] loading models...")
cnn_fer = to_device(ResNet(1, len(emotion_classes)), get_default_device())
cnn_fer.load_state_dict(torch.load(model_path + 'CNN_fer.pth')['state_dict'])
cnn_fer.eval()

cnn_ck = to_device(ResNet(1, len(emotion_classes)), get_default_device())
cnn_ck.load_state_dict(torch.load(model_path + 'CNN_ck.pt')['state_dict'])
cnn_ck.eval()

def start():
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    c, d = 0, 0
    label_ck, label_fer = '', ''
    while True:
        image = vs.read()
        rects = detector(image, 0)

        if not len(rects):
            cv2.imshow("Frame", image)
            continue

        rect = rects[0]
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)              
            
        (x,y,w,h) = face_utils.rect_to_bb(rect) # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x,y,w,h)]
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) # draw the face bounding box

        ## graylevels
        dim = 48
        crop = image[y:y+h, x:x+w]
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop = cv2.equalizeHist(crop)
        crop = resize(crop, dim)
        Image = crop.reshape(1,1,dim,dim) / 255.0
##        Image = crop / 255.0
        
##        dim = 48
##        cnn_input = gray[y:y+h, x:x+w] # gray is shape(480,640)
##        cnn_input = resize(cnn_input)
##        cnn_input = imutils.resize(cnn_input, width=int(dim*1.05)) # buffer of 5 pixels for cropping to 100x100
##        cnn_input = cnn_input[:dim,:dim]
##        cnn_input = cnn_input.reshape(1,1,dim,dim)/255.0 # shape=(1,1,dim,dim)
        
        # prediction
        with torch.no_grad():
            pred_tensor_ck = cnn_ck(torch.Tensor(Image).cuda()).argmax()
            pred_ck = pred_tensor_ck.cpu().numpy().item()
            pred_tensor_fer = cnn_fer(torch.Tensor(Image).cuda()).argmax()
            pred_fer = pred_tensor_fer.cpu().numpy().item()

        # stabilisation
        if not c:
            pred_label_ck = emotion_classes[pred_ck]
            c += 1
        elif c == 5:
            label_ck = pred_label_ck
            c = 0
        else:
            pred_label_ck = emotion_classes[pred_ck] if (pred_label_ck == emotion_classes[pred_ck]) else ''            
            c = c+1 if pred_label_ck else 0

        if not d:
            pred_label_fer = emotion_classes[pred_fer]
            d += 1
        elif d == 5:
            label_fer = pred_label_fer
            d = 0
        else:
            pred_label_fer = emotion_classes[pred_fer] if (pred_label_fer == emotion_classes[pred_fer]) else ''            
            d = d+1 if pred_label_fer else 0
        
        # show frame
##        cv2.putText(image, label_ck, (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
##        cv2.putText(image, label_fer, (x,y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)        
        frame = imutils.resize(crop.reshape(dim,dim,1),400)
        cv2.putText(frame, label_ck, (5,400-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(frame, label_fer, (5,400-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imshow("Frame", frame)

        # break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

start()    
