import cv2

WINDOW = "I'm listening!"

cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
img = cv2.imread('WALL-E-Listening.png')
imgS = cv2.resize(img, (384, 240))
cv2.imshow(WINDOW, imgS)
cv2.waitKey(6000)
