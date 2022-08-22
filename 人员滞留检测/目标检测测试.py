from ElectronicFence import ElectronicFence
from utils.tools import letterbox
import os
import cv2
import numpy as np
ef = ElectronicFence()
img = cv2.imread("./images/2.jpeg")
im, ratio, (dw, dh),new_unpad = letterbox(img)
print("最终的hw的填充值是：{},{}".format(dh,dw))
# cv2.imshow("rsource_img",img)
# cv2.imshow("letterbox_result",im)
# cv2.waitKey(0)
if not os.path.exists("./images/letterbox_output.jpeg"):
    print("保存letterbox的输出图像")
    cv2.imwrite("./images/letterbox_output.jpeg",im)
print("letterbox输出图像的shape_hwc:{}".format(im.shape))
alarm,result_image = ef.get_result(img,[[[200, 230], [730, 230], [730, 680], [200, 680]]])
print(alarm)