import cv2
import numpy as np
import random
import time
from VideoCapture import VideoCapture

def SimonSays_nothing(selected_item,cam):

    WINDOW = "Simon Says"

#    items_of_selection = ["bottle","sports ball","cell phone","spoon","fork"]
#    selected_item = random.choice(items_of_selection)

    lst = ["rectangular object","cylindrical object","spherical object"]
    choose = selected_item
    classes = []

    sample = []

    if choose not in lst:
        yolo = cv2.dnn.readNet('./object_data/yolov4.weights' , './object_data/yolov4.cfg')
        with open("./object_data/coco.names","r") as f:
            classes = f.read().splitlines()
    else:
        yolo = cv2.dnn.readNet('./object_data/iteration3_final.weights' , './object_data/customyolov4-obj.cfg')
        with open("./object_data/obj.names","r") as f:
            classes = f.read().splitlines()

    starting_time = time.time()

    width, height = 640, 480

#    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
#    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_cnt = 0
    while(True):
        frame = cam.read()
        frame_cnt += 1

        blob = cv2.dnn.blobFromImage(frame,1/255, (320,320), (0,0,0), swapRB=True, crop=False)
        yolo.setInput(blob)
        output_layer_names = yolo.getUnconnectedOutLayersNames()
        layeroutput = yolo.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layeroutput:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2] *width)
                    h = int(detection[3]*height)

                    x = int(center_x-w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

        color = (0, 0, 255)

        cv2.putText(frame, "Please find this item: " + choose, (0,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size=(len(boxes),3))
        print('[%d]Detected:' % frame_cnt, end="")
        first_item = 1
        if len(indexes) > 0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]

                label = str(classes[class_ids[i]])
                confi = str(round(confidences[i],2))
                print(('' if first_item else ','), label, end="")
                first_item = 0

                if label not in sample:
                    sample.append(label)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame, label + " " + confi, (x,y+20), font, 2, (255,255,255), 2)
        else:
            print(" none", end="")

        if choose in sample:
#            print("\nI didn't say Simon Says! Listen carefully next time :)")
            cv2.putText(frame, "Wrong!", (180,240), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 6)
            ans = False
            break
#            cv2.imshow(WINDOW, frame)
#            cv2.waitKey(1)
#            choose = random.choice(items_of_selection)
#            sample = []
#            continue
        elif time.time() - starting_time > 10:
#            print("\nTime's up! You won!")
            cv2.putText(frame, "Well Done!", (130,240), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 6)
            ans = True
            break

        # Display the frame.
#        cv2.imshow(WINDOW, frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

        print(" "*100, end="\r") # padding

    print()
#    cv2.imshow(WINDOW, frame)
#    cv2.waitKey(1)
#    time.sleep(2)
#    cv2.destroyAllWindows()
    return ans

#cam = VideoCapture(0)
#SimonSays_nothing("cell phone", cam)
#cam.release()
