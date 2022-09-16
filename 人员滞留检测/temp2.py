import cv2
video_cap = cv2.VideoCapture("./videos/1.mp4")
count = 0
while video_cap.isOpened():
    ret,frame = video_cap.read()
    count += 1
    if count%64 == 0:
        cv2.imwrite("./videos/sample{}.jpeg".format(count),frame)
