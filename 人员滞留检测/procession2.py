import os
import glob
import cv2
import shutil
import datetime

import numpy as np

from connection import *
from apscheduler.schedulers.blocking import BlockingScheduler
from matching_the_person import MatchPerson
from detector import Detector

#判断是否处于时间点1
def time1(camera_name,dir_path="./person_images/detections")->bool:
    images = os.listdir(dir_path)
    for image in images:
        if image.startswith(camera_name+"_time1"):
            return True
    return False

def generate_error_image(camera_name,camera_rtsp,date:str="error",h:int=600,w:int=1400):
    background = np.ones((h,w,3),dtype=np.uint8)*255
    date_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    error_text = "Error occurs when getting video from {}".format(camera_rtsp)
    cv2.imshow("back",background)
    cv2.waitKey(0)

generate_error_image()

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
        pass
    if time1(camera_name=camera_name):
        pass

def blocking_scheduler(task,dt,mp,start_date="",interval=30):
    '''
    :param task: 任务处理函数
    :param dt: 检测模型
    :param mp: 匹配模型
    :param interval: seconds,waited
    :return:
    '''
    camera_info= get_source()
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    #不同的摄像头任务属于不同的线程，某个任务出错不会导致整个程序停止
    for camera in camera_info:
        scheduler.add_job(task, 'interval',start_date=start_date,args=[camera,dt,mp], seconds=interval, max_instances=1)
    scheduler.start()







