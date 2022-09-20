from detector import Detector
from model_utils.tools import letterbox
import os
import cv2
from connection import get_one_frame
import numpy as np
ef = Detector()
def from_rtsp():
    rtsp_uri = "rtsp://127.0.0.1:8554/video1"
    image = get_one_frame(rtsp_uri)
    area = [[[161, 59], [1319, 45], [1269, 765], [127, 735], [161, 59]]]
    # area = np.array(area,np.int32)
    # cv2.polylines(image, area, isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    # cv2.namedWindow('area', 0)
    # cv2.resizeWindow('area', 1000, 800)
    # cv2.imshow("area",image)
    # cv2.waitKey(0)
    ef.crop_the_person_out(image, area, camera_name="camera0", time_point="time2",
                           saved_path="./temp_images/detections")

def from_local_image():
    image = cv2.imread("./temp_images/2.jpeg")
    prediction,_ = ef.inference(image)
    print(prediction)
from_local_image()