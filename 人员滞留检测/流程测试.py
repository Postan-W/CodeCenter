from detector import Detector
from matching_the_person import MatchPerson
from connection import get_one_frame
from procession import load_images,filter_the_result,get_unique,get_origin,datetime_clear,delete_camera_images
import os
import cv2
import numpy as np

def detections(time_point):
    dt = Detector()
    rtsp_uri = "rtsp://127.0.0.1:8554/video1"
    image = get_one_frame(rtsp_uri)
    area = [[[221,91],[1321,81],[161,755],[129,727],[221,91]]]
    dt.crop_the_person_out(image, area, camera_name="camera0", time_point=time_point,
                           saved_path="./temp_images/detections")
# detections("time2")
def do_matching(camera_name,mp:MatchPerson,result_path="./temp_images/results"):
    '''
    :param camera_name: 摄像头名称
    :param mp:  特征匹配类对象
    :param result_path: 匹配结果的图片保存位置
    :return:
    '''
    time1_images,time1_image_name = load_images(camera_name,time_point="time1",image_dir="./temp_images/detections")
    time2_images,time2_image_name = load_images(camera_name,time_point="time2",image_dir="./temp_images/detections")
    print(len(time1_images),len(time2_images))
    """
    time1中不同的图片可能会匹配到time2中的同一个x，因为time2中的x是time2图片里距离它们最近的,time1中这些和x相似的图片(即欧氏距离小于阈值)中，选取
    距离最小的那个作为真正的匹配者，其余的被舍弃。
    """
    time1_similar_index = []
    time2_similar_index = []
    time2_similar_min_distance = []
    for i,img in enumerate(time1_images):
        most_likely_index,min_distance,all_targets = mp.matching_persons(img,time2_images)
        if not most_likely_index == -1:
            shape = img.shape
            img2 = cv2.resize(time2_images[most_likely_index],(shape[1],shape[0]))
            temp = np.hstack((img,img2))
            cv2.namedWindow("hsta", 0)
            # cv2.resizeWindow("hsta", 600, 1300)
            cv2.imshow("hsta", temp)
            cv2.waitKey(0)
            time1_similar_index.append(i)
            time2_similar_index.append(most_likely_index)
            time2_similar_min_distance.append(min_distance)
    print(time1_similar_index,time2_similar_index)
    index_to_reserve = filter_the_result(time2_similar_index,time2_similar_min_distance)#过滤掉匹配到同一个目标的那种情况
    print(index_to_reserve)
    time1_similar_index_reserve = list(np.array(time1_similar_index)[index_to_reserve])
    time2_similar_index_reserve = list(np.array(time2_similar_index)[index_to_reserve])
    time1_image_name = list(np.array(time1_image_name)[time1_similar_index_reserve])
    time2_image_name = list(np.array(time2_image_name)[time2_similar_index_reserve])
    #得到原始frame
    time1_origin_frame = get_origin(time1_image_name,img_dir="./temp_images/detections")
    time2_origin_frame = get_origin(time2_image_name,img_dir="./temp_images/detections")
    # #将结果写入文件
    count = 0
    for compared in list(zip(time1_origin_frame,time2_origin_frame)):
        count += 1
        vstack = np.vstack((compared[0],compared[1]))
        cv2.imwrite(os.path.join(result_path,camera_name+datetime_clear()+"_"+str(count)+".jpeg"),vstack)
do_matching("camera0",MatchPerson(em_ckpt_file="weights/net_MobileNet_IFN.pth"))