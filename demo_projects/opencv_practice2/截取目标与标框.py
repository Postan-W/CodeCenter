import cv2
import time
#需要注意的是，cv2对象是numpy ndarray,维度为hwc。但描述像素点的位置还是(x,y)即宽高的形式，所以注意和cv2对象的宽高对应
img_path = "./images/lenaNoise.jpeg"
img = cv2.imread(img_path)
shape = img.shape
#定义左上与右下点
up_left,down_right = (round(shape[1]*0.3),round(shape[0]*0.2)),(round(shape[1]*0.7),round(shape[0]*0.8))
cut_out = img[up_left[1]:down_right[1],up_left[0]:down_right[0]]
# cv2.imwrite("target.jpeg",cut_out)
cv2.imshow("cut_out",cut_out)
cv2.waitKey(0)
#框出目标
line_thick = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1#长和宽平均值的1/500,加一是为了避免值为0
#参数依次是cv2图片对象、左上角(x,y)、右下角(x,y)、框的颜色、粗细、框线的类型
cv2.rectangle(img, up_left, down_right, [0,0,255], thickness=line_thick, lineType=cv2.LINE_AA)
cv2.imshow("rect",img)
cv2.waitKey(0)