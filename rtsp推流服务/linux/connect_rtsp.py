import cv2
import sys
print(sys.argv[1])
rtsp_uri = "rtsp://localhost:18554/stream{}".format(sys.argv[1])
cap = cv2.VideoCapture(rtsp_uri)
if cap.isOpened():
    print(rtsp_uri,"连接成功")