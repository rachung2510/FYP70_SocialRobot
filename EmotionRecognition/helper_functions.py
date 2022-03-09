import math, imutils, torch
import numpy as np

# Find vector magnitude
def mag(pointA, pointB):
    x = pointA[0] - pointB[0]
    y = pointA[1] - pointB[1]
    return math.sqrt(x*x + y*y)

# Find angle between two points (-pi to pi rads)
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

# Resize image to square with given dimensions
def resize(img, dim):
    w, h = img.shape[1], img.shape[0]
    (length, width) = (w,h) if w>h else (h,w)
    factor = 50 / width
    img = imutils.resize(img, width=math.ceil(w*factor))
    return img[:dim, :dim]

# Use GPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

##def getEmotionClass(probArr, emotion_classes):
##    prob = np.sum(probArr, axis=0)
##    pred = np.argmax(probArr, axis=1)
##    strs = ['CNN (vec)','SVM','CNN (px1)','CNN (px2)']
##    for i in range(len(strs)):
##        print('%s:' % strs[i], emotion_classes[pred[i]], end=", ")
##    print(emotion_classes[np.argmax(prob)].upper())        
##    return np.argmax(prob)
