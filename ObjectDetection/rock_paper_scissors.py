import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
 
 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4,2)


choices = [0,2,5]
computer_choice = random.choice(choices)

starting_time = time.time()

video = cv2.VideoCapture(0)


#status 0 -> Lose
#status 1 -> Win
#status 2 -> Draw
def game_time (upCount,computer_choice):

    start = time.time()
    while(time.time() - start < 5):
        if computer_choice == 0: #computer stone
            if upCount == 2: #person scissors
                status =  0 #Lose
            elif upCount == 5: #person paper
                status =  1 #win
            elif upCount == 0: #person stone
                status =  2 #draw

        elif computer_choice == 2: #computer scissors
            if upCount == 2: #person scissors
                status =  2 #draw
            elif upCount == 5: #person paper
                status =  0 #Lose
            elif upCount == 0: #person stone
                status =  1 #win

        elif computer_choice == 5: #computer paper
            if upCount == 2: #person scissors
                status =  1 #Win
            elif upCount == 5: #person paper
                status =  2 # draw
            elif upCount == 0: #person stone
                status =  0 #Lose
    return status





 
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while video.isOpened():
        _, frame = video.read()
 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        image = cv2.flip(image, 1)
         
        imageHeight, imageWidth, _ = image.shape
 
        results = hands.process(image)
   
 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,0,255)


        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )

 
 
        if results.multi_hand_landmarks != None:

            handList = []
            for handLandmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, handLandmarks, mp.solutions.hands.HAND_CONNECTIONS)
                for idx, lm in enumerate(handLandmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    handList.append((cx, cy))
                for point in handList:
                    cv2.circle(frame, point, 10, (255, 255, 0), cv2.FILLED)
                upCount=0
                for coordinate in finger_Coord:
                    if handList[coordinate[0]][1] < handList[coordinate[1]][1]:
                        upCount += 1
                if handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]:
                    upCount += 1
                print(upCount)

        cv2.imshow('Hand Tracking', image)
        #time.sleep(1)
 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
 
video.release()
cv2.destroyAllWindows()




'''
                game_status = game_time(upCount,computer_choice)
                if game_status == 0: #Lose
                    cv2.putText(frame, "You Lost!"  , (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                elif game_status==1: #win
                    cv2.putText(frame, "You Won!"  , (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                elif game_status==2: #Draw
                    cv2.putText(frame, "Its a Draw!"  , (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            #computer_choice = random.choice(choices)


                        cv2.putText(frame, "Computer's Choice: " + str(computer_choice), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

'''