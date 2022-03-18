import cv2
import numpy as np
import random
import time


def SimonSays_nothing(selected_item,cam):
    lst = ["rectangular object","cylindrical object","spherical object"]
    choose = selected_item
    classes = []

    sample = []

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

    starting_time = time.time()

    width, height = 640, 480

#    items_of_selection = ["bottle","sports ball","spoon","fork"]
#    items_of_selection = ["bottle","sports ball","cell phone"]
#    selected_item = random.choice(items_of_selection)


    while(True):
        frame = cam.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#        if time.time() - starting_time > 30:
#            break

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

        #print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

        color = (0, 0, 255)

        cv2.putText(frame, "Please find this Item: " + choose, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size=(len(boxes),3))
        if len(indexes) >0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]

                label = str(classes[class_ids[i]])
                confi = str(round(confidences[i],2))
                #color = colors[i]
                if label not in sample:
                    sample.append(label)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame, label + " " + confi, (x,y+20), font, 2, (255,255,255), 2)

        if time.time() - starting_time > 10:
            if choose not in sample:
                cv2.putText(frame, "Well Done!!!", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                ans = True
            else:
                cv2.putText(frame, "Wrong!!!", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                ans = False
            break
        # print(sample)

        cv2.imshow("Video",frame)

    return ans

#cam = USBCamera(capture_device=0, width=640, height=480)
#SimonSays_nothing("cell_phone", cam)
#cam.cap.release()
#cv2.destroyAllWindows()
