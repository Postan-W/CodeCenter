import cv2
import os

urls = [
        "rtsp://admin:hik12345@10.25.27.212:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.213:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.215:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.216:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.217:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.218:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.219:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.220:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.221:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.211:554/h264/ch1/main/av_stream",
        "rtsp://admin:hik12345@10.25.27.214:554/h264/ch1/main/av_stream",
        ]

logs = open("./logs.txt","a",encoding='utf-8')
for i,url in enumerate(urls):
    try:
        cap = cv2.VideoCapture("mask.mp4")
        ret, frame = cap.read()
        cv2.imwrite(str(i) + ".jpg", frame)
        cap.release()
        cv2.destroyAllWindows()
        logs.write("请求的地址是:{}".format(url)+"\n")
    except Exception as e:
        logs.write("--------------------出错-----------------------"+"\n")
        logs.write(e+"\n")
        logs.write("--------------------出错-----------------------"+"\n")
