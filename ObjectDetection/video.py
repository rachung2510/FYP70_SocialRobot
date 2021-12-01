import cv2
import matplotlib.pyplot as plt
import numpy as np


# ./yolov4.weights ./darknet-master/cfg/yolov4.cfg
yolo = cv2.dnn.readNet('./yolov4.weights' , './yolov4.cfg')
#yolo = cv2.dnn.readNet('./yolov4.weights' , './yolov4-tiny.cfg')
classes = []
with open("./coco.names","r") as f:
    classes = f.read().splitlines()
print((classes))

cam = cv2.VideoCapture(0)
if (cam.isOpened() == False):
    print("Unable to read camera feed")
width = int(cam.get(3))
height = int(cam.get(4))
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))

while(True):
    ret,frame = cam.read()
    if ret == True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not frame.any():
        print("Nothing")
        break
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


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))
    if len(indexes) >0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]

            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i],2))
            color = colors[i]

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame, label + " " + confi, (x,y+20), font, 2, (255,255,255), 2)

            #plt.imshow(img)
            #plt.show()

    cv2.imshow("Video",frame)

cam.release()
#out.release()

cv2.destroyAllWindows()