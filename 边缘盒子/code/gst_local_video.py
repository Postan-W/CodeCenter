#-*- coding:utf-8 -*-
import cv2
pipeline = "filesrc location=/home/nvidia/wmztemp/code/mask.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink "
capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
capture = cv2.VideoCapture("./mask.mp4",cv2.CAP_FFMPEG)


