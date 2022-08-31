import cv2
import numpy as np

area = [[[29,246],[374,20],[653,90],[468,420]]]
area = np.array(area,np.int32)
image = cv2.imread("./images/cat.jpeg")
cv2.polylines(image, area, isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.namedWindow('area', 0)
cv2.resizeWindow('area', 600, 500)
cv2.imshow("area",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

