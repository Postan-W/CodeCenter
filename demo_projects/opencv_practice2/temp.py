import cv2
rtsp_url = "rtsp://39.96.162.75:8554/stream1"
cap = cv2.VideoCapture(rtsp_url)
print(cap.isOpened())
# while cap.isOpened():
#     ret,frame = cap.read()
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     cv2.imshow("对冲", frame)