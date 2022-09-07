#!/usr/bin/python
from procession import *
from public_logger import logger

if __name__ == "__main__":
    dt = Detector()  # 检测
    mp = MatchPerson(em_ckpt_file="weights/net_MobileNet_IFN.pth")  # 匹配
    logger.info("模型加载完成")
    blocking_scheduler(camera_task,dt,mp,start_date=start_date(2),interval=40)
    logger.info("任务启动")