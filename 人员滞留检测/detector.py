# -*- coding: utf-8 -*-
# 2022/08
import torch
import cv2
import os
import numpy as np
from model_utils.tools import letterbox, non_max_suppression, scale_coords, in_poly_area, plot_one_box, np_to_str, draw_poly_area
import random
from public_logger import logger
#本类中所有函数的代码都是基于一张图片的推理
#本类实现了人体检测；是否进入电子围栏的判断；目标人体截取等方法
class Detector(object):
    def __init__(self, device=",".join([str(i) for i in range(torch.cuda.device_count())]), model="./weights/yolobest.pt"):
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else 'cpu')
        # 加载检测模型
        self.model = torch.load(model, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

        self.confthre = 0.6
        self.nmsthre = 0.45
        self.img_size = 640

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.label = ["person"]#只识别人

    def inference(self, image):
        img = letterbox(image, new_shape=self.img_size)[0]#接收的是单张图片
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR到RGB;hwc到chw
        img = np.ascontiguousarray(img)#转为元素内存连续数组，提高数据处理速度
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()#uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0) #在0轴处插入一维，即batch维
        pred = self.model(img, augment=False)[0]#推理
        pred = non_max_suppression(pred, self.confthre, self.nmsthre, classes=None, agnostic=False)#NMS筛选框
        return pred, img

    def crop_the_person_out(self,image,area,camera_name,time_point,saved_path="./person_images/detections/"):
        '''
        :param image: 从rtsp流中取到的图片
        :param area: 检测区域[[[x1,y1],[x2,y2]...]...]
        :param camera_name: 摄像头名称
        :param time_point: 检测时间点
        :return:
        '''
        predictions,img = self.inference(image)
        count = 0
        #虽然写作for循环，实际上只有一张图片的推理结果
        for i, det in enumerate(predictions):
            if len(det):#如果该图片存在检测的目标框
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()#坐标取整
                for *xyxy, conf, cls in reversed(det):#原始输出按照预测框的conf值从高到低排序，这里是按照conf值从低到高排列
                    if self.names[int(cls)] in self.label:#如果是人，就把ta裁剪出来
                        if in_poly_area(xyxy, area):#采用几何方法判断目标中心点是否在划定区域内
                            image_temp = image.copy()#这样做是为了避免不同目标的框在同一张图上出现
                            person = image_temp[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                            #把该人在原图上框出来
                            cv2.imwrite(os.path.join(saved_path,"{}_{}_{}.jpeg".format(camera_name,time_point,count)),person)
                            plot_one_box(xyxy, image_temp, color=(0, 255,0))
                            cv2.imwrite(os.path.join(saved_path,"{}_origin{}_{}.jpeg".format(camera_name,time_point,count)),image_temp)
                            count += 1




#area_point_list = [[[200, 230], [730, 230], [730, 680], [200, 680]]]

