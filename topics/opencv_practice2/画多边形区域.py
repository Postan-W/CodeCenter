import cv2
import numpy as np

area = [[[29,246],[374,5],[653,90],[468,420],[29,246]]]
area = np.array(area,np.int32)
image = cv2.imread("./images/cat.jpeg")
"""
多边形区域是从列表中的第一个点到最后一个点一次连接的。
isClosed参数代表是否强制图形为封闭的：
isClosed=True时，如果第一个点和最后一个点不是一样的，以为着区域不封闭，那么函数会将二者连接起来形成封闭图形；而isClosed=False时保持图形原样
"""
cv2.polylines(image, area, isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.namedWindow('area', 0)
cv2.resizeWindow('area', 600, 500)
cv2.imshow("area",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

