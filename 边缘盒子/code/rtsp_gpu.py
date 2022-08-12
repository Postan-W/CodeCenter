#-*- coding:utf-8 -*-
import cv2
uri = "rtsp://admin:bonc123456@172.16.67.250:554/h264/ch1/main/av_stream"
# uri = "rtsp://admin:bonc123456@172.16.67.250:554"
video_path = "./testgpu.avi"
#cv2通过GSTREAMER获取流
def open_cam_rtsp(uri, width, height, latency):
    '''
    :param uri:RTSP URI, e.g. rtsp://192.168.1.64:554
    :param width:image width [1920]
    :param height:image height [1080]
    :param latency:latency in ms for RTSP [200 default]
    :return:
    '''
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    print(gst_str)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def use_gpu(uri,video_path):
    w = 1280
    h = 720
    vid_cap = open_cam_rtsp(uri=uri,width=w,height=h,latency=200)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    fourcc = 'XVID'
    vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*fourcc), 25, (w, h))

    while vid_cap.isOpened():
        print("已连接成功")
        ret, frame = vid_cap.read()
        if not ret:
            print("连接错误")
            break
        vid_writer.write(frame)

    vid_cap.release()

use_gpu(uri=uri,video_path=video_path)
