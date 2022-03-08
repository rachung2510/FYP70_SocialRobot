import math
import numpy as np

## Functions for finding vector magnitude and direction
def mag(pointA, pointB):
    x = pointA[0] - pointB[0]
    y = pointA[1] - pointB[1]
    return math.sqrt(x*x + y*y)

# find angle between two points (-pi to pi rads)
def angle(cog, point):
    x = point[0] - cog[0]
    y = point[1] - cog[1]
    
    if not x:
        return math.pi/2 if y>0 else -math.pi/2
        
    angle = math.atan(y/x)
    if x<0 and y>0: # 2nd quadrant
        angle += math.pi
    elif x<0 and y<0: # 3rd quadrant
        angle -= math.pi
    return angle

def getEmotionClass(probArr, emotion_classes):
    prob = np.sum(probArr, axis=0)
    pred = np.argmax(probArr, axis=1)
    strs = ['CNN (vec)','SVM','CNN (px1)','CNN (px2)']
    for i in range(len(strs)):
        print('%s:' % strs[i], emotion_classes[pred[i]], end=", ")
    print(emotion_classes[np.argmax(prob)].upper())        
    return np.argmax(prob)
