#from jetcam.usb_camera import USBCamera
from VideoCapture import VideoCapture
import cv2
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dev", type=int, nargs="?", default=0)
args = parser.parse_args()

WINDOW = "Cam Test"
cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#cam = USBCamera(capture_device=args.dev, width=640, height=480)
cam = VideoCapture(args.dev, width=640, height=480)
while True:
    frame = cam.read()
    cv2.imshow(WINDOW, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

