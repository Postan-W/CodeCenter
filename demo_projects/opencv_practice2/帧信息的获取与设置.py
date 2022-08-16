#video.set(cv2.CAP_PROP_POS_FRAMES, number)用来指定接下来要处理的帧的位置
import cv2
video = cv2.VideoCapture("./videos/hotelCalifornia.mp4")
fps = video.get(5)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter("./videos/hotelCalifornia_passframe.mp4",cv2.VideoWriter_fourcc('X','2','6','4'),fps,size)
print("帧率是:{}".format(fps))
#获取总帧数
total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
print("总帧数是:{}".format(total_frames))
print("帧数除以帧率就是视频的大致时间:{}".format(round(total_frames/fps,1)))
count = 0
#每隔30帧读取，相当于快进
while video.isOpened():
    count += 30
    video.set(cv2.CAP_PROP_POS_FRAMES, count)
    ret,frame = video.read()
    video_writer.write(frame)