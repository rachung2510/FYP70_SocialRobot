import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
from jetcam.usb_camera import USBCamera


def enemy(image, x_enemy, y_enemy):
  #x_enemy=random.randint(50,600)
  #y_enemy=random.randint(50,400)
  cv2.circle(image, (x_enemy,y_enemy), 25, (0, 200, 0), 5)
  #score=score+1

def PopTheBubble(video):

    WINDOW = "Pop the Bubble!"

    start = time.time()
    score = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    x_enemy =random.randint(50,600)
    y_enemy=random.randint(50,400)

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
            font=cv2.FONT_HERSHEY_SIMPLEX
            color=(255,0,255)
            text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
            text=cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)

            enemy(image, x_enemy, y_enemy)

            if score == 10:
                print("Your time taken is: ",time.time() - start)
                status = 1
                duration = time.time() - start
                time.sleep(3)
                break

            if time.time() - start >= 60:
                print("Sorry, Time out")
                status = 0
                duration = 999
                time.sleep(3)
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

                        point=str(point)
                        #print(point)
                        if point=='HandLandmark.INDEX_FINGER_TIP':
                            try:
                                cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 25, (0, 200, 0), 5)
                                #print(pixelCoordinatesLandmark[1])
                                if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0]==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                                    #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                                #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                                    print("found")
                                    x_enemy=random.randint(50,600)
                                    y_enemy=random.randint(50,400)
                                    score=score+1
                                    font=cv2.FONT_HERSHEY_SIMPLEX
                                    color=(255,0,255)
                                    text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                    enemy(image, x_enemy, y_enemy)
                            except:
                                pass

            cv2.imshow(WINDOW, image)
#           time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q') :
#                print(score)
                break

    cv2.destroyAllWindows()
    return (status, duration)


#video = USBCamera(capture_device=0, width=640, height=480)
#(status, duration) = PopTheBubble(video)
#print(status, duration)
#video.cap.release()
#cv2.destroyAllWindows()
