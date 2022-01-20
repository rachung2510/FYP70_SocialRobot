# Merged Object Learning Branch
Combines the object detection and emotion prediction parts and outputs a video capture displaying both predictions.

Main file ***object_learning.py*** loads the models, starts the default webcam and calls ***video.py*** from **ObjectDetection/** and ***realtime_emotion_recognition.py*** from **EmotionRecognition/** to obtain the predictions.
1. ```cd FYP70_SocialRobot\```.
2. Run ```python object_learning.py``` in your terminal.
