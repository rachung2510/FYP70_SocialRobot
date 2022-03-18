from jetcam.usb_camera import USBCamera
import cv2
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dev", type=int, nargs="?", default=0)
args = parser.parse_args()

cam = USBCamera(capture_device=args.dev, width=640, height=480)
while True:
    frame = cam.read()
    cv2.imshow("Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.cap.release()
cv2.destroyAllWindows()

