import os
import glob
import cv2
import datetime
import numpy as np
from connection import *
from public_logger import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from matching_the_person import MatchPerson
from detector import Detector

start_date = lambda delta=0:(datetime.datetime.now()+datetime.timedelta(seconds=delta)).strftime('%Y-%m-%d %H:%M:%S')
datetime_clear = lambda :datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ","").replace("-","").replace(":","")

#判断是否处于时间点1
def time1(camera_name,dir_path="./person_images/detections")->bool:
    images = os.listdir(dir_path)
    for image in images:
        if image.startswith(camera_name+"_time1"):
            return False
    return True

def delete_camera_images(camera_name,images_dir="./person_images/detections/"):
    image_path = glob.glob(images_dir+"/*")
    image_name = os.listdir(images_dir)
    for i,image in enumerate(image_name):
        if image.startswith(camera_name):
            os.remove(image_path[i])

def load_images(camera_name:str,time_point:str,image_dir="./person_images/detections/")->(list,list):
    '''
    :param camera_name: 摄像头名称
    :param time_point:  时间点
    :return: 两个列表。cv2加载的图片，即元素为numpy.ndarray；图片的名称。
    '''
    image_path = glob.glob(os.path.join(image_dir,camera_name+"_"+time_point+"*"))
    image_name = []#也要保存图片的名字，用于匹配后对应上原像
    for image in image_path:
        image_name.append(os.path.split(image)[1])
    image_loaded = []
    for image in image_path:
        image_loaded.append(cv2.imread(image))

    return image_loaded,image_name

def generate_error_image(camera_name,camera_rtsp,h:int=600,w:int=1400,save_path="./person_images/results"):
    '''
    本函数仅用来生成错误提示图片，用于在连接摄像头失败时。
    :param camera_name: 摄像头名称
    :param camera_rtsp: 摄像头rtsp地址
    :param h: 提示图片的高
    :param w: 提示图片的宽
    :param save_path: 提示图片的保存位置
    :return:
    '''
    background = np.ones((h,w,3),dtype=np.uint8)*255
    date_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_text_font, date_text_scale = cv2.FONT_HERSHEY_COMPLEX, 2
    date_text_color = (10, 20, 20)
    date_text_position = (int(w / 32), int(h / 8))
    date_text_thickness = 3
    cv2.putText(background,date_text,date_text_position,date_text_font,date_text_scale,date_text_color,date_text_thickness)
    error_text = "Error occurs when getting video from {}".format(camera_rtsp)
    error_text_font, error_text_scale = cv2.FONT_HERSHEY_COMPLEX, 0.8
    error_text_color = (0, 0, 255)
    error_text_position = (int(w / 32), int(h / 2))
    error_text_thickness = 2
    cv2.putText(background,error_text,error_text_position,error_text_font,error_text_scale,error_text_color,error_text_thickness)
    image_name = camera_name+"error"+datetime_clear()+".jpeg"
    cv2.imwrite(os.path.join(save_path,image_name),background)

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

def get_origin(source:list,img_dir="./person_images/detections"):
    origin_img_name = []
    for img in source:
        temp = img.split("_")
        origin_img_name.append("{}_origin{}_{}".format(temp[0],temp[1],temp[2]))
    origin_frame = []
    for img in origin_img_name:
        origin_frame.append(cv2.imread(os.path.join(img_dir,img)))
    return origin_frame

def do_matching(camera_name,mp:MatchPerson,result_path="./person_images/results"):
    '''
    :param camera_name: 摄像头名称
    :param mp:  特征匹配类对象
    :param result_path: 匹配结果的图片保存位置
    :return:
    '''
    time1_images,time1_image_name = load_images(camera_name,time_point="time1")
    time2_images,time2_image_name = load_images(camera_name,time_point="time2")
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
            # shape = img.shape
            # img2 = cv2.resize(time2_images[most_likely_index],(shape[1],shape[0]))
            # temp = np.hstack((img,img2))
            # cv2.namedWindow("hsta", 0)
            # # cv2.resizeWindow("hsta", 600, 1300)
            # cv2.imshow("hsta", temp)
            # cv2.waitKey(0)
            time1_similar_index.append(i)
            time2_similar_index.append(most_likely_index)
            time2_similar_min_distance.append(min_distance)
    index_to_reserve = filter_the_result(time2_similar_index,time2_similar_min_distance)#过滤掉匹配到同一个目标的那种情况
    time1_similar_index_reserve = list(np.array(time1_similar_index)[index_to_reserve])
    time2_similar_index_reserve = list(np.array(time2_similar_index)[index_to_reserve])
    time1_image_name = list(np.array(time1_image_name)[time1_similar_index_reserve])
    time2_image_name = list(np.array(time2_image_name)[time2_similar_index_reserve])
    #得到原始frame
    time1_origin_frame = get_origin(time1_image_name)
    time2_origin_frame = get_origin(time2_image_name)
    # #将结果写入文件
    count = 0
    for compared in list(zip(time1_origin_frame,time2_origin_frame)):
        count += 1
        vstack = np.vstack((compared[0],compared[1]))
        cv2.imwrite(os.path.join(result_path,camera_name+"_"+datetime_clear()+"_"+str(count)+".jpeg"),vstack)

    #删除两个时间点的该摄像头的所有图片
    delete_camera_images(camera_name)

def camera_task(camera_info:tuple,dt:Detector,mp:MatchPerson):
    '''
     :param camera_info: (camera_name:str,(rtsp:str,area:[[[],...],...]))
     :param dt: 检测模型
     :param mp: 匹配模型
     :return:
     '''
    camera_name, camera_rtsp, area = camera_info[0], camera_info[1][0], camera_info[1][1]
    frame = get_one_frame(camera_rtsp)
    if len(frame) == 0:
        generate_error_image(camera_name,camera_rtsp)
        return
    elif time1(camera_name=camera_name):
        dt.crop_the_person_out(frame,area,camera_name,time_point="time1")
        logger.info("{}第一个时间点结束".format(camera_name))
    else:
        dt.crop_the_person_out(frame, area, camera_name, time_point="time2")
        #开始进行匹配
        logger.info("{}第二个时间点结束".format(camera_name))
        do_matching(camera_name,mp)
        logger.info("{}匹配完毕".format(camera_name))
        #do_matching删除了两个时间点的图片后立即开始新的检测
        frame = get_one_frame(camera_rtsp)
        if len(frame) == 0:
            generate_error_image(camera_name, camera_rtsp)
            return
        dt.crop_the_person_out(frame, area, camera_name, time_point="time1")
        logger.info("{}第一个时间点结束".format(camera_name))

def blocking_scheduler(task,dt,mp,start_date="",interval=30):
    '''
    :param task: 任务处理函数
    :param dt: 检测模型
    :param mp: 匹配模型
    :param interval: seconds,waitting for next task
    :return:
    '''
    camera_info= get_source()
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    #不同的摄像头任务相互独立
    for camera in camera_info:
        scheduler.add_job(task, 'interval',start_date=start_date,args=[camera,dt,mp], seconds=interval, max_instances=1)
    scheduler.start()







