import torch
from torch import nn
from imutils import face_utils
import imutils, time, pickle, math
import dlib, cv2
import numpy as np
from VideoCapture import VideoCapture

# define constants
model_path = 'emotion_data/'
emotion_classes = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
WINDOW = "Emotion Recognition"

def init_emotion():
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path+ 'shape_predictor_68_face_landmarks.dat')

    print("[INFO] loading emotion models...")
    fcnn_fer = FCNNModel(68*4, 128, 7) # input, hidden, output
    fcnn_fer.load_state_dict(torch.load(model_path + 'FCNN_norm_128_fer.pt', map_location='cpu')['state_dict'])
    fcnn_fer.eval()

    return detector, predictor, (fcnn_fer,)

def get_emotion_class(frame, detector, predictor, models, display=False):
    fcnn_fer = models[0]
    emotion_class = "neutral"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detector(gray, 0)

    face_area = 0
    main_rect = None
    for (i, rect) in enumerate(rects):
        # get bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if display:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # show the face number

        # proceed only for main face (largest area)
        if w*h >= face_area:
            face_area = w*h
            main_rect = rect
            rx, ry, rw, rh = x, y, w, h

    if not main_rect:
        return emotion_class

    vectors, coords = [], []
    shape = predictor(gray, main_rect)
    shape = face_utils.shape_to_np(shape)

    # extract features
    cog = tuple(shape.mean(axis=0).astype(int)) # get center of gravity (COG)
    for (x,y) in shape:
        vectors.append([mag(cog, (x,y)), angle(cog, (x,y))]) # get vector magnitude and direction
        coords.append([x,y]) # append coordinates relative to cog
        if display:
            cv2.line(frame, (x,y), cog, (255,0,0), 1) # draw vector lines
            cv2.circle(frame, (x,y), 1, (0,0,255), -1) # draw markers
            cv2.circle(frame, cog, 5, (255,255,0), -1) # draw center of gravity

    # input
    vectors = np.array(vectors)
    vectors[:,0] /= max(vectors[:,0]) # normalize magnitudes
    vectors[:,1] = (vectors[:,1] + math.pi) / (2*math.pi) # normalize direction
    vectors = vectors.reshape(-1)
    coords = np.array(coords)
    coords -= np.c_[min(coords[:,0]), min(coords[:,1])]
    coords = coords / np.c_[max(coords[:,0]), max(coords[:,1])]
    coords = coords.reshape(-1)
    Vector = np.dstack((vectors, coords)).reshape(1, -1) # shape=(samples, vectors+coords)

    # prediction
    with torch.no_grad():
        pred_tensor_fer = fcnn_fer(torch.Tensor(Vector)).argmax()
        pred_fer = pred_tensor_fer.cpu().numpy().item()
        emotion_class = emotion_classes[pred_fer]

    if display:
        cv2.putText(frame, emotion_class.upper(), (rx-20,ry+rh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow(WINDOW, frame)

    return emotion_class

##-------------------FUNCTIONS-----------------------##

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

##-------------------CLASSES-----------------------##

class FCNNModel(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, num_classes):
        super(FCNNModel, self).__init__()

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



#vs = VideoCapture(0)
#detector, predictor, models = init_emotion()
#cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#while True:
#    frame = vs.read()
#    get_emotion_class(frame, detector, predictor, models, display=True)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cv2.destroyAllWindows()
#vs.release()
