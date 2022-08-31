from configparser import ConfigParser
import cv2
import numpy as np

def get_source(datatype="rtsp",config_path="./sources.ini",keyname="rtsp_uri",encoding="utf-8")->list:
    conf = ConfigParser()
    conf.read(config_path, encoding=encoding)
    if datatype == "rtsp":
        return list(conf[keyname].values())
    else:
        pass

#获取检测时的那一时刻的一张图片,如果没有成功获取那一时刻的图片，做了20次上限的容错
def get_one_frame(uri:str)->np.ndarray:
    try:
        video = cv2.VideoCapture(uri)
        if not video.isOpened():
            raise Exception
        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                return frame
            else:
                if count == 20:
                    return None
                else:
                    count += 1
    except:
        return None

