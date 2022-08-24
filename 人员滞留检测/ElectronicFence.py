# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 17:17
# @Author  : Chuck

import torch
import cv2
import numpy as np
from utils.tools import letterbox, non_max_suppression, scale_coords, in_poly_area, plot_one_box, np_to_str, draw_poly_area


class ElectronicFence(object):
    """
    电子围栏
    """
    def __init__(self, device=0, weight="./weights/best.pt"):
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else 'cpu')
        # Load model
        self.model = torch.load(weight, map_location=self.device)['model'].float()  # load to FP32
        self.model.to(self.device).eval()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

        self.confthre = 0.6
        self.nmsthre = 0.45
        self.img_size = 640

        # Get names and colors
        """
        这里列出所有的name:
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
         'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
          'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        """
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # self.label = ["person", "car", "truck", "bicycle", "motorbike"]#这里面定义要检测的目标类型
        self.label = ["person"]

    def inference(self, image):
        # Run inference
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)# init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        # Padded resize
        img = letterbox(image, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, and hwc to chw
        img = np.ascontiguousarray(img)#转为元素内存连续数组，使得下游计算更快

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()#uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) #在0处插入一维，即batch维

        # Inference
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.confthre, self.nmsthre, classes=None, agnostic=False)

        return pred, img

    def detections(self, pred, img, image, area_point_list):
        draw_poly_area(image, area_point_list)
        alarm = False
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],image.shape).round()#round即将小数舍入为整数
                for *xyxy, conf, cls in reversed(det):#原始输出的每个框置信度是从高到低排列的，这里是把置信度从低到高排列
                    if self.names[int(cls)] in self.label:
                        if in_poly_area(xyxy, area_point_list):
                            alarm = True
                            plot_one_box(xyxy, image, [0, 0, 255])
        alarm_image = np_to_str(image)

        return alarm, alarm_image

    def get_result(self, image, area_point_list):
        pred, img = self.inference(image)
        alarm, image = self.detections(pred, img, image, area_point_list)

        return alarm, image



if __name__ == '__main__':
    ab_C = './test.jpg'
    image = cv2.imread(ab_C)

    area_point_list = [[[200, 230], [730, 230], [730, 680], [200, 680]]]
    EF = ElectronicFence()
    alarm, image = EF.get_result(image, area_point_list)
    print(alarm)
