import json
import numpy as np
from connection import get_one_frame,get_source
import cv2
import os
# camera_info= get_source()
# area = np.array(camera_info[1][1][1],np.int32)
# frame = get_one_frame(camera_info[1][1][0])
# cv2.polylines(frame,area,isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
# cv2.imshow("frame",frame)
# cv2.waitKey(0)
# cv2.imwrite("./temp_images/video1.jpeg",frame)
frame = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
if frame == "":
    print("frame不存在")
print(len(np.array([])))
print(os.path.exists("./person_images/camera0"))