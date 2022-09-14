from configparser import ConfigParser
import cv2
import numpy as np
import json
from public_logger import logger

def get_cuda_number(config_path="./conf/config1.ini"):
    conf = ConfigParser()
    conf.read(config_path)
    cuda_info = dict(conf['CUDA'])
    return cuda_info['cuda_visible_devices']

def get_source(datatype="rtsp",config_path="./conf/config1.ini",encoding="utf-8")->tuple:
    conf = ConfigParser()
    conf.read(config_path, encoding=encoding)
    if datatype == "rtsp":
        rtsps = list(conf["rtsp_uri"].values())
        rtsp_dict = {"camera{}".format(i):(rtsp.split("+")[0],json.loads(rtsp.split("+")[1])) for i,rtsp in enumerate(rtsps)}
        return tuple(rtsp_dict.items())
    else:
        pass

#获取检测时的那一时刻的一张图片,如果没有成功获取那一时刻的图片，做了3次上限的容错
def get_one_frame(uri:str)->np.ndarray:
    try:
        video = cv2.VideoCapture(uri)
        if not video.isOpened():
            logger.info("{}没有连接上".format(uri))
            raise Exception
        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                video.release()
                return frame
            else:
                if count == 3:
                    return np.array([])
                else:
                    count += 1
    except:
        return np.array([])

# print(get_one_frame("rtsp://127.0.0.1:8554/video2"))