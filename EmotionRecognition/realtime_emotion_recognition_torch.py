from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import torch
from torch import nn
import numpy as np
from helper_functions import mag, angle

# define constants
model_path = 'trained_models/'
emotion_classes = ['anger','disgust','fear','happiness','neutral','sadness','surprise']

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load models
print("[INFO] loading models...")
class FCNNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FCNNModel, self).__init__()
        
        input_layer_size = kwargs['input_layer_size']
        hidden_layer_size = kwargs['hidden_layer_size']
        num_classes = kwargs['num_classes']
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
fcnn_model = torch.load(model_path + 'FCNN_model.pt')    

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

        vectors = np.array(vectors)        
        scale_factor = 1 / max(vectors[:,0])
        vectors[:,0] = vectors[:,0] * scale_factor # normalize magnitudes
        coords = np.array(coords) * scale_factor
        coords = coords.reshape(-1)
        
        # prediction
        fcnn_input = np.dstack((vectors.reshape(1, -1), coords)).reshape(1, -1) # shape=(1, 272)
        if fcnn_input.shape == (1,272):
            with torch.no_grad():
                pred_tensor = fcnn_model(torch.Tensor(fcnn_input).cuda()).argmax()        
                pred = pred_tensor.cpu().numpy().item()
            pred_label = emotion_classes[pred]
            cv2.putText(image, '%s' % pred_label, (x,y+h+20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)    
                    
        cv2.imshow("Frame", image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

start()    
