from imutils import face_utils
import imutils
import cv2
import numpy as np
from helper_functions import mag, angle, getEmotionClass

dim = 50
def get_emotion(frame, detector, predictor, models):
    # store vectors as input data for model prediction
    vectors, coords = [], []
    classes = {}
    emotion_classes = ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
    cnn2, svm2, cnnA, cnnB = models[0], models[1], models[2], models[3]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    rects = detector(gray, 0) # detect faces in the grayscale frame

    face_area = 0
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect) # determine facial landmarks for face region
        shape = face_utils.shape_to_np(shape) # convert facial landmark (x,y)-coords to Numpy array

        # extract features
        cog = tuple(shape.mean(axis=0).astype(int)) # get center of gravity (COG)
        for (x,y) in shape:
            vectors.append([mag(cog, (x,y)), angle(cog, (x,y))]) # get vector magnitude and direction
            coords.append([x-cog[0], y-cog[1]])

        # get bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect) # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x,y,w,h)]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) # draw the face bounding box
        cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # show the face number
        
        # model inputs
        vectors = np.array(vectors)
        scale_factor = 1 / max(vectors[:,0])
        vectors[:,0] = vectors[:,0] * scale_factor # normalize magnitudes
        vectors = vectors[:,0] * vectors[:,1] # get product of magnitude & direction
        coords = (np.array(coords) * scale_factor).reshape(-1) # 1D array

        cnn3_input = np.zeros((1, len(vectors)+len(coords), 1))
        svm2_input = np.zeros((1, len(vectors)+len(coords)))
        # actions for main face (largest area)
        if w*h >= face_area:
            face_area = w*h # store new max face_area

            # store inputs
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
            cnn_px_input = imutils.resize(cnn_px_input, width=int(dim*1.1)) # buffer for cropping
            cnn_px_input = np.expand_dims(cnn_px_input[:dim,:dim], axis=0) # shape=(1,dim,dim)

            # draw facial landmark features on image
            for (xx,yy) in shape:
                cv2.line(frame, (xx,yy), cog, (255,0,0), 1) # draw vector lines
                cv2.circle(frame, (xx,yy), 1, (0,0,255), -1) # draw markers
            cv2.circle(frame, cog, 5, (255,255,0), -1) # draw center of gravity

    # prediction
    classes['CNN2'] = emotion_classes[np.argmax(cnn2.predict(cnn_v_input))]
    classes['SVM2'] = emotion_classes[svm2.predict(svm_vc_input)[0]]
    classes['CNNA'] = emotion_classes[np.argmax(cnnA.predict(cnn_px_input))]
    classes['CNNB'] = emotion_classes[np.argmax(cnnB.predict(cnn_px_input))]

    cnn2_prob = cnn2.predict(cnn_v_input)
    svm2_prob = 2 * svm2.predict_proba(svm_vc_input)
    cnnA_prob = 0.5 * cnnA.predict(cnn_px_input)
    cnnB_prob = 0.5 * cnnB.predict(cnn_px_input)
    emotion_class = emotion_classes[getEmotionClass(np.r_[cnn2_prob, svm2_prob, cnnB_prob, cnnB_prob], \
                                                    emotion_classes)]
    classes['FINAL'] = emotion_class

    cv2.putText(frame, emotion_class.upper(), (x-20,y+h+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return frame
