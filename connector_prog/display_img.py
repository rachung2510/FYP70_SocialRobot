import cv2

cv2.namedWindow("I'm listening!", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
img = cv2.imread('WALL-E-Listening.png')
imgS = cv2.resize(img, (384, 240))
cv2.imshow("I'm listening!",imgS)
cv2.waitKey(6000)