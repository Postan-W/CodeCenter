# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 17:12
# @Author  : Chuck
from .StaffLeave import StaffLeave
from .ElectronicFence import ElectronicFence
from .PersonGathered import PersonGathered
from .ImageEnhance import Enhance

class SecurityServer(object):
    """
    视频安防模型调用类
    """
    def __init__(self):
        self.SL = StaffLeave()
        self.EF = ElectronicFence()
        self.PG = PersonGathered()
        self.ImageEnhance = Enhance()


    def staff_leave(self, image, alarm_nums, area_point_list):
        """
        人员离岗
        :param image: numpy格式图像
        :param alarm_nums: 报警值，即要求几人在岗
        :param area_point_list: 监控区域像素点列表
        :return: alarm: True or False True为报警,False为不报警
                 image: 报警图像,若报警则会标记人员矩形框，否则为原图
        """
        image = self.image_enhance(image)
        alarm, image = self.SL.get_result(image, alarm_nums, area_point_list)

        return alarm, image

    def person_gathered(self, image, alarm_nums, area_point_list):
        """
        人员聚集
        :param image: numpy格式图像
        :param alarm_nums: 报警值，即要求超过几人为聚集
        :param area_point_list: 监控区域像素点列表
        :return: alarm: True or False True为报警,False为不报警
                 image: 报警图像,若报警则会标记人员矩形框，否则为原图
        """
        image = self.image_enhance(image)
        alarm, image = self.PG.get_result(image, alarm_nums, area_point_list)

        return alarm, image

    def electronic_fence(self, image, area_point_list):
        """
        电子围栏
        :param image: numpy格式图像
        :param area_point_list: 监控区域像素点列表
        :return: alarm: True or False True为报警,False为不报警
                 image: 报警图像,若报警则会标记人员或车辆矩形框，否则为原图
        """
        image = self.image_enhance(image)
        alarm, image = self.EF.get_result(image, area_point_list)

        return alarm, image
    #图像增强算法，非调度任务
    def image_enhance(self, image):
        image = self.ImageEnhance(image)
        return image
