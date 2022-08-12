import cv2
video = cv2.VideoCapture("./videos/california_short.mp4")
fps = video.get(5)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter("./videos/hotelCalifornia_gray.mp4",cv2.VideoWriter_fourcc('X','2','6','4'),fps,size)
while video.isOpened():
    ret, frame = video.read()
    if ret:
        print(ret)

