
***realtime_emotion_recognition.py***: Main file to perform realtime emotion prediction using the default webcam and models trained from *model.ipynb*. For each detected face in the frame:
  - Bounding box in green
  - Facial landmarks markers as red dots
  - Center marker (COG) as cyan dot
  - Vectors of landamrks to COG as blue lines
  - Predicted emotion classes by different trained models

***helper_function.py***: Functions to get the magnitude and angle of the vector of two points. Called by *realtime_emotion_recognition.py*.

***shape_predictor_68_face_landmarks.dat***: Pretrained facial landmark detector from dlib. Used by *realtime_emotion_recognition.py* to find the coordinates of the facial landmarks.

***model.ipynb***: Used for training and evaluation of different models. Satisfactory models are stored in *trained_model/* folder.

***model_descriptions.txt***: Brief overview of the trained models, including model type, params, input features (vectors or coords?), and performance (accuracy/loss).

***dataset/***: Contains the CK+ dataset of 980 images for 8 emotions: anger, contempt, dsgust, fear, happiness, neutral, sadness and surprise.

***trained_model/***: Stores the trained models to be called for prediction by *realtime_emotion_recognition.py*.
