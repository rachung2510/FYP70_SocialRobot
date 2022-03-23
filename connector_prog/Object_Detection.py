import cv2
import numpy as np
import time
from jetcam.usb_camera import USBCamera
import random
from VideoCapture import VideoCapture

def SimonSays_item(selected_item, cam):

    WINDOW = "Simon Says"

    starting_time = time.time()
    width, height = 640, 480

    lst = ["rectangular object","cylindrical object","spherical object"]
    classes = []
    choose = selected_item

    if choose not in lst:
        yolo = cv2.dnn.readNet('./object_data/yolov4.weights' , './object_data/yolov4.cfg')
        with open("./object_data/coco.names","r") as f:
            classes = f.read().splitlines()
        # print((classes))
    else:
        yolo = cv2.dnn.readNet('./object_data/iteration3_final.weights' , './object_data/customyolov4-obj.cfg')
        with open("./object_data/obj.names","r") as f:
            classes = f.read().splitlines()
        # print((classes))

#    items_of_selection = ["bottle","sports ball","spoon","fork"]
    items_of_selection = ["bottle","sports ball","cell phone","backpack"]
#    selected_item = random.choice(items_of_selection)

    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while(True):

        frame = cam.read()

        if (time.time() - starting_time) > 60:
            print("\nTime's Up!")
            print('Time taken: %.2f' % (time.time() - starting_time))
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)
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

        #print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

        color = (0, 0, 255)

#        cv2.putText(frame, "Please find this Item: " + selected_item, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size=(len(boxes),3))
        print('Detected: ', end="")
        if len(indexes) > 0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]

                label = str(classes[class_ids[i]])
                confi = str(round(confidences[i],2))
                #color = colors[i]
                print(label, end=", ")
                if label == selected_item:
                    print('\nMatch!')
                    print('Time taken: %.2f' % (time.time() - starting_time))
                    color = (0, 255, 0)
#                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
#                    time.sleep(2)
#                    start = time.time()
#                    while (time.time() - start > 5000000):
#                        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
#                    selected_item = random.choice(items_of_selection)
                    status_level=0
#                    cv2.destroyAllWindows()
                    return True

#                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
#                cv2.putText(frame, label + " " + confi, (x,y+20), font, 2, (255,255,255), 2)

#        cv2.imshow(WINDOW, frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

        print(" "*100, end="\r")

#    cv2.destroyAllWindows()
    return False

def get_time(prev):
    curr = time.time()
    print("Time:", curr - prev)
    return curr

#cam = VideoCapture(1)
#SimonSays_item("backpack", cam)
#cam.release()
#cv2.destroyAllWindows()
