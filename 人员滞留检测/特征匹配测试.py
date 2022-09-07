from matching_the_person import MatchPerson
import cv2
import glob
import numpy as np
mp = MatchPerson(em_ckpt_file="weights/net_MobileNet_IFN.pth")
image_path = glob.glob("./temp_images/trial_images/*")
image_list = []
for image in image_path:
    image_list.append(cv2.imread(image))

source_image = image_list[0]
most_likely_index,distance,all_targets = mp.matching_persons(source_image,image_list)
temp = np.vstack((source_image,image_list[most_likely_index]))
cv2.namedWindow('most_likely', 0)
cv2.resizeWindow('most_likely', 600, 1300)
cv2.imshow("most_likely",temp)
cv2.waitKey(0)
print("所有相似的图片个数为{},索引为{}:".format(len(all_targets),all_targets))
for index in all_targets:
    name = 'index{}'.format(index) if not index == most_likely_index else "most_likely"
    temp = np.vstack((source_image, image_list[index]))
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 600, 1300)
    cv2.imshow(name, temp)
    cv2.waitKey(0)
# image1 = cv2.imread("person_images/time1/camera1_0.jpeg")
# image2 = cv2.imread("person_images/time1/camera1_1.jpeg")
# image3 = cv2.imread("person_images/time1/camera1_2.jpeg")
# image4= cv2.imread("person_images/time1/camera1_0.jpeg")
# image_array = [image1,image2,image3,image4]
# image1_embedding = mp.encode(image1)
# image2_embedding = mp.encode(image2)
# print(image1_embedding)
# print(image2_embedding)
# #原代码里设定的阈值是0.15，小于这个值认为是同一个目标
# print(mp.euclidean_dist(image1_embedding,image2_embedding))
# print(mp.more_encode(image_array))
# print(mp.matching_persons(image1,image_array))