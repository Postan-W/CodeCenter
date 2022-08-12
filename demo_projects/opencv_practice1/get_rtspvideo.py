import cv2
import os
rtsp_urls = ["rtsp://admin:hik12345@10.25.27.212:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.213:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.215:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.216:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.217:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.218:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.219:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.220:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.221:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.211:554/h264/ch1/main/av_stream",
             "rtsp://admin:hik12345@10.25.27.214:554/h264/ch1/main/av_stream"]

for url in rtsp_urls:
    vid_cap = cv2.VideoCapture(url)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    fourcc = 'XVID'
    size = (w, h)
    vid_writer = cv2.VideoWriter("{}.avi".format(url[32:34]), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    try:
        #这里限制了视频的大小为10M,超过就不再继续保存，然后释放该连接，进行下一个连接的视频保存
        while vid_cap.isOpened() and int(os.path.getsize("{}.avi".format(url[32:34]))/1024/1024) < 10:
            ret, frame = vid_cap.read()
            if not ret:
                continue
            vid_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    vid_cap.release()