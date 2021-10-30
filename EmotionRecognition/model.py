from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to grayscale
    frame = vs.read()   
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # show the face number
        cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # get center of gravity (COG)
        cog = tuple(shape.mean(axis=0).astype(int))
        
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x,y) in shape:
            cv2.line(frame, (x,y), cog, (255,0,0), 1) # draw vector lines
            cv2.circle(frame, (x,y), 1, (0,0,255), -1) # draw markers
            
        cv2.circle(frame, cog, 5, (255,255,0), -1) # draw center of gravity
                
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
