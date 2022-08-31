from matching_the_person import MatchPerson
import cv2
mp = MatchPerson(em_ckpt_file="weights/net_MobileNet_IFN.pth")
image1 = cv2.imread("person_images/time1/passagers_0.jpeg")
image2 = cv2.imread("person_images/time1/passagers_1.jpeg")
image3 = cv2.imread("person_images/time1/passagers_2.jpeg")
image4= cv2.imread("person_images/time1/passagers_0.jpeg")
image_array = [image1,image2,image3,image4]
image1_embedding = mp.encode(image1)
image2_embedding = mp.encode(image2)
# print(image1_embedding)
# print(image2_embedding)
# #原代码里设定的阈值是0.15，小于这个值认为是同一个目标
print(mp.euclidean_dist(image1_embedding,image2_embedding))
# print(mp.more_encode(image_array))
print(mp.matching_persons(image1,image_array))