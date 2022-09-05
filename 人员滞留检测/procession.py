import os
import glob
import cv2
import shutil
import datetime
from connection import *
from apscheduler.schedulers.blocking import BlockingScheduler
from matching_the_person import MatchPerson
from detector import Detector

#删除文件夹及其下面的所有文件
def delete_image_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

#提示：下面所有的filtering_name参数用于过滤图片是多余的，因为写入图片方式有变，不同摄像头的图片放在摄像头对应的文件夹里，不用根据摄像头过滤
#获取特定摄像头的检测图片
def get_images(dir_path,filtering_name)->list:
    '''
    :param dir_path: 图片所在路径
    :param filtering_name: 按名称筛选
    :return:
    '''
    dir_path = dir_path+"/" if not dir_path.endswith("/") else dir_path
    images = glob.glob(dir_path + filtering_name + "*")
    return images

#加载图片
def cv2_load_image(dir_path,filtering_name)->list:
    image_path_list = get_images(dir_path,filtering_name)
    image_loaded = []
    for img in image_path_list:
        image_loaded.append(cv2.imread(img))

    return image_loaded

# image_loaded = cv2_load_image(get_images("./person_images/detections1","camera0"))
#删除文件夹下的所有图片
def delete_image(dir_path,filtering_name):
    image_path_list = get_images(dir_path,filtering_name)
    for img in image_path_list:
        os.remove(img)

def move_file_or_dir(src,dst):
    shutil.move(src,dst)

#定时任务开始运行的时间。delta是当前时间之后的秒数
start_date = lambda delta=0:(datetime.datetime.now()+datetime.timedelta(seconds=delta)).strftime('%Y-%m-%d %H:%M:%S')
datetime_clear = lambda :datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ","").replace("-","").replace(":","")

#匹配两个文件夹下的所有图片
def do_matching(dir1,dir2,dir_origin1,dir_origin2,camera_name,mp:MatchPerson,result_path="./person_images/results"):
    dir1_imgs = cv2_load_image(dir1,camera_name)
    dir2_imgs = cv2_load_image(dir2,camera_name)
    dir_origin1_imgs_path_list = get_images(dir_origin1,camera_name)
    dir_origin2_imgs_path_list = get_images(dir_origin2,camera_name)
    dir1_img_num = len(dir1_imgs)
    dir2_img_num = len(dir2_imgs)
    if dir1_img_num == 0 or dir2_img_num == 0:
        logger.info("{}时间点1或2没有检测到人".format(camera_name))
        return
    else:
        logger.info("{}时间点1人数为:{},时间点2人数为:{}".format(camera_name,dir1_img_num,dir2_img_num))
        logger.info("开始匹配...")
        start = datetime.datetime.now()
        for i,img in enumerate(dir1_imgs):
            target_index = mp.matching_persons(img,dir2_imgs)
            if not target_index == -1:
                time_stamp = datetime_clear()
                result_name_time1 = camera_name+"_"+time_stamp+"_time1.jpeg"
                result_name_time2 = camera_name+"_"+time_stamp+"_time2.jpeg"
                move_file_or_dir(dir_origin1_imgs_path_list[i],os.path.join(result_path,result_name_time1))
                move_file_or_dir(dir_origin2_imgs_path_list[target_index],os.path.join(result_path,result_name_time2))
        end = datetime.datetime.now()
        logger.info("{}匹配完成，总耗时:{}".format(camera_name,end-start))
        delete_image_dir(dir1)
        delete_image_dir(dir2)
        delete_image_dir(dir_origin1)
        delete_image_dir(dir_origin2)


"""
周期:
一个周期包含两个时间点，第一个时间点获取目标，存放在camera_x_1文件夹下，并将对应的原图存在camera_x_origin_1下；
第二个时间点获取目标，存放在camera_x_2文件夹下，并将对应的原图存在camera_x_origin_2下；第二个时间点结束，开始匹配
camera_x_1和camera_x_2下的图片,将匹配结果存在./person_images/result下面，从命名方式上进行匹配。
匹配完后将上述camera_x有关的所有文件夹删除。一个周期结束。
"""
def camera_task(camera,dt:Detector,mp:MatchPerson):
    '''
    :param camera: (camera_name:str,(rtsp:str,area:[[[],...],...]))
    :param dt: 检测模型
    :param mp: 匹配模型
    :return:
    '''
    camera_name, camera_rtsp, area = camera[0], camera[1][0], camera[1][1]
    tag = 0
    time1_path = "./person_images/" + camera_name + "_1"
    time1_origin_path = "./person_images/" + camera_name + "_origin_1"
    time2_path = "./person_images/" + camera_name + "_2"
    time2_origin_path = "./person_images/" + camera_name + "_origin_2"
    if not os.path.exists(time1_path):
        os.mkdir(time1_path)
        os.mkdir(time1_origin_path)
        tag = 1
        logger.info("{}生成time1文件夹".format(camera_name))
    else:
        os.mkdir(time2_path)
        os.mkdir(time2_origin_path)
        tag = 2
    frame = get_one_frame(camera_rtsp)
    if len(frame) == 0:
        return
    if tag == 1:
        dt.crop_the_person_out(frame,area,image_name=camera_name,person_saved_path=time1_path,origin_saved_path=time1_origin_path)
    elif tag == 2:
        dt.crop_the_person_out(frame, area,image_name=camera_name,person_saved_path=time2_path,
                               origin_saved_path=time2_origin_path)
        do_matching(time1_path,time2_path,time1_origin_path,time2_origin_path,camera_name,mp)




#一个摄像头开启一个定时任务，对于某一个摄像头，前一个任务不完成，后一个任务不开始
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



# delete_image("./person_images/time1","camera0")





