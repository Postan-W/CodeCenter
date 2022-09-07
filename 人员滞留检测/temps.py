import os.path

import cv2
import numpy as np
l1 = [1,2,3,4,5,6,7,8]
l2 = [12, 34, 54,12,34,  34,  78, 112]
l3 = [0.8,0.9,23,45,0.34,0.11,2.2,6.7]
print(np.array(l2)[[0,3,4]])
results = list(zip(l1,l2))
print(np.array(l2)==34)
def get_unique(source_list):
    result = []
    for element in source_list:
        if not element in result:
            result.append(element)
    return result

#查找一个列表中两个相同的元素，并根据第二个列表的条件进行过滤,返回要保留的元素的index
def filter_the_result(source:list,condition:list):
    #注意len(source)==len(condition)
    unique_list = get_unique(source)
    source = np.array(source)
    all_equal_list = []
    for num in unique_list:
        equals = source == num
        equal_list = []
        for i,equal in enumerate(equals):
            if equal:
                equal_list.append(i)
        all_equal_list.append(equal_list)

    index_to_reserve = []
    for ele in all_equal_list:
        index_to_reserve.append(ele[np.argmin(np.array(condition)[ele])])
    return sorted(index_to_reserve)


print(filter_the_result(l2,l3))
def get_index_target(source:list,index:list):
    return list(np.array(source)[index])

def get_origin(source:list,img_dir="./person_images/detections"):
    origin_img_name = []
    for img in source:
        temp = img.split("_")
        origin_img_name.append("{}_origin{}_{}".format(temp[0],temp[1],temp[2]))
    origin_frame = []
    for img in origin_img_name:
        origin_frame.append(cv2.imread(os.path.join((img_dir,img))))
    return origin_frame

source = ["camera0_time1_1.png","camera0_time1_1.png"]
print(get_origin(source))


