import cv2
rtsp_url = "rtsp://localhost:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret,frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("对冲", frame)