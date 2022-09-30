from person_detection import Detector
import cv2
import numpy as np

detector = Detector()
image = cv2.imread("./trial/source_images/1.jpeg")

detector.person_detecting(image,image_name="first")
