import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
from VideoCapture import VideoCapture

def PopTheBubble(video):

    WINDOW = "Pop the Bubble!"

    start = time.time()
    score, status = 0, 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    x_enemy = random.randint(50,600)
    y_enemy = random.randint(50,400)

    # Create named window for resizing purposes.
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while True:
            frame = video.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            imageHeight, imageWidth, _ = image.shape
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255,0,255)
            cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
            cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)

            cv2.circle(image, (x_enemy,y_enemy), 25, (0, 255, 0), 5)

            if score == 10:
                cv2.putText(image, "You Win!", (200,240), font, 2, (0,255,0), 8)
                status = 1
                duration = time.time() - start
                print("Your time taken is: ", duration)
                break

            if time.time() - start >= 60:
                print("Sorry, Time out")
                cv2.putText(image, "Time out :(", (140,240), font, 2, (0,0,255), 8)
                status = 0
                duration = 999
                break

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )

            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in mp_hands.HandLandmark:

                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        if str(point) == 'HandLandmark.INDEX_FINGER_TIP':
                            try:
                                cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 25, (0, 255, 0), 5)
                                if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0]==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                                    x_enemy = random.randint(50,600)
                                    y_enemy = random.randint(50,400)
                                    score += 1
                                    cv2.putText(frame, "Score", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4, cv2.LINE_AA)
                                    cv2.circle(image, (x_enemy,y_enemy), 25, (0, 255, 0), 5)
                            except:
                                pass

            cv2.imshow(WINDOW, image)

            if cv2.waitKey(1) & 0xFF == ord('q') :
                duration = time.time() - start
                break

    cv2.imshow(WINDOW, image)
    cv2.waitKey(1)
    time.sleep(3)
    cv2.destroyAllWindows()
    return (status, duration)


#video = USBCamera(capture_device=0, width=820, height=480)
#video = VideoCapture(0)
#(status, duration) = PopTheBubble(video)
#video.release()
