from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import torch
import numpy as np
from helper_functions import mag, angle
from torch_classes import FCNNModel, ResNet, to_device, get_default_device

# define constants
model_path = 'trained_models/'
emotion_classes = ['anger','disgust','fear','happiness','neutral','sadness','surprise']

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load models
print("[INFO] loading models...")
##fcnn_model = torch.load(model_path + 'FCNN_model.pt')
cnn_model = to_device(ResNet(1, len(emotion_classes)), get_default_device())
cnn_model.load_state_dict(torch.load(model_path + 'CNNModel.pth'))
cnn_model.eval()

def start():
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        vectors, coords = [], []

        image = vs.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detector(gray, 1)         

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)        

            cog = tuple(shape.mean(axis=0).astype(int)) # get center of gravity (COG)            
            for (x,y) in shape:
                cv2.line(image, (x,y), cog, (0,0,255), 1) # draw vector lines
                cv2.circle(image, (x,y), 2, (255,0,0), -1) # image, center-coords, radius, colour, thickness(fill)
                vectors.append([mag(cog, (x,y)), angle(cog, (x,y))]) # get vector magnitude and direction                
                coords.append([x-cog[0], y-cog[1]]) # append coordinates relative to cog
                cv2.circle(image, cog, 5, (0,255,255), -1)    
                
            (x,y,w,h) = face_utils.rect_to_bb(rect) # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x,y,w,h)]
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) # draw the face bounding box
            cv2.putText(image, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # show face number

        ## vectors
        vectors = np.array(vectors)        
        scale_factor = 1 / max(vectors[:,0])
        vectors[:,0] = vectors[:,0] * scale_factor # normalize magnitudes
        coords = np.array(coords) * scale_factor
        coords = coords.reshape(-1)

        ## graylevels
        dim = 50
        cnn_input = image[y:y+h, x:x+w] # crop to face
        cnn_input = cv2.cvtColor(cnn_input, cv2.COLOR_RGB2GRAY) # convert to grayscale
        cnn_input = cv2.equalizeHist(cnn_input) # equalize histogram
        cnn_input = imutils.resize(cnn_input, width=int(dim*1.05)) # buffer of 5 pixels for cropping to 100x100
        cnn_input = cnn_input[:dim,:dim]
        cnn_input = cnn_input.reshape(1,1,dim,dim)/255.0 # shape=(1,1,dim,dim)
        
        # prediction
##        fcnn_input = np.dstack((vectors.reshape(1, -1), coords)).reshape(1, -1) # shape=(1, 272)
##        if fcnn_input.shape == (1,272):
##            with torch.no_grad():
##                fcnn_pred_tensor = fcnn_model(torch.Tensor(fcnn_input).cuda()).argmax()
##                fcnn_pred = fcnn_pred_tensor.cpu().numpy().item()                
##            fcnn_pred_label = mood_map[fcnn_pred]
##            cv2.putText(image, '%s' % fcnn_pred_label, (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        with torch.no_grad():
            cnn_pred_tensor = cnn_model(torch.Tensor(cnn_input).cuda()).argmax()
            cnn_pred = cnn_pred_tensor.cpu().numpy().item()
        cnn_pred_label = mood_map[cnn_pred]            
        cv2.putText(image, '%s' % cnn_pred_label, (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)    
                    
        cv2.imshow("Frame", image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

start()    
