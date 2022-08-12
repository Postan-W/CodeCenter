import cv2
import time
#或者简写为：rtsp://admin:bonc123456@172.16.67.250
url = "rtsp://admin:bonc123456@172.16.67.250:554/h264/ch1/main/av_stream"
video_path = "./testcpu.avi"
def use_cpu(url,video_path):
    vid_cap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    print((w, h), fps)
    fourcc = 'XVID'
    vid_writer = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    while vid_cap.isOpened():
        ret, frame = vid_cap.read()
        if not ret:
            print("连接错误")
            break

        vid_writer.write(frame)

    vid_cap.release()

# use_cpu(url=url,video_path=video_path)

vid_cap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
print(vid_cap.isOpened(),vid_cap.read())
