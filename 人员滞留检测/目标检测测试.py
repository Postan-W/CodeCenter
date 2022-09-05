from detector import Detector
from model_utils.tools import letterbox
import os
import cv2
import numpy as np
ef = Detector()
image = cv2.imread("temp_images/2.jpeg")
area = [[[89,437],[1949,456],[2009,1925],[69,1841],[89,437]]]
# area = np.array(area,np.int32)
# cv2.polylines(image, area, isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
# cv2.namedWindow('area', 0)
# cv2.resizeWindow('area', 1000, 800)
# cv2.imshow("area",image)
# cv2.waitKey(0)
ef.crop_the_person_out(image, area, image_name="camera1", person_saved_path="person_images/time1", frame_saved_path=
"person_images/detections1")
