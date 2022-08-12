import cv2
pipeline = "rtspsrc location=\"rtsp://admin:bonc123456@172.16.67.250:554/h264/ch1/main/av_stream\" ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! \"video/x-raw, format=(string)BGRx! videoconvert\" ! appsink"

pipeline2 = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){},'
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format("rtsp://{}:{}@{}:{}".format("admin","bonc123456","172.16.67.250",554),200,1280,720)

print(pipeline)
print(pipeline2)
capture = cv2.VideoCapture(pipeline,cv2.CAP_GSTREAMER)

pipeline = "rtspsrc location=\"rtsp://admin:bonc123456@172.16.67.250:554/h264/ch1/main/av_stream\" ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx! videoconvert ! appsink"

