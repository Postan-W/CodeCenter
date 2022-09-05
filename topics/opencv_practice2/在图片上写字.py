import cv2
import numpy as np
h = 600
w = 1400
text = "this is the test text"
font,font_size= cv2.FONT_HERSHEY_COMPLEX,2
color = (10,20,20)
position = (int(w/4),int(h/2))
thickness = 3
background = np.ones((600,1400,3),dtype=np.uint8)*255
cv2.putText(background,text,position,font,font_size,color,thickness)
cv2.imshow("text",background)
cv2.waitKey(0)