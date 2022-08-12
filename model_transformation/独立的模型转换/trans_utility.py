import re,json
import numpy as np
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
#创建日志对象
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s || filename=%(filename)s || function=%(funcName)s || line=%(lineno)d || information=%(message)s',datefmt="%y-%m-%d %H:%M:%S")
#输出到文件。maxBytes的单位即为byte,所以5*1024*1024即为5Mb
file_handler = RotatingFileHandler("./logs/log.txt", maxBytes=5*1024*1024 , backupCount=100)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
#输出到屏幕
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console)

dir_dict = {1: "./models/", 2: "./intermediate_models/", 3: "./transformed_models/"}
model_dict = {"Pytorch":["pkl","pt","pth"],"Keras":["h5"],"Tensorflow":["pb"]}

def h5_input_shape(model_json:str,logger=None)-> list:
    model_structure = json.loads(model_json)
    # 通过反向预查来匹配模型输入
    pattern = re.compile('(?<=\'batch_input_shape\': ).*?\]')
    #上面的正则的意思依次是：反向预查，匹配除“\n”和"\r"之外的任何单个字符，任意多次，非贪婪模式(即把每次匹配的机会先让给后面),以]结束
    result = pattern.search(str(model_structure["config"])).group()
    pattern2 = re.compile("\d+?(?=,|\])")
    shape = pattern2.findall(result)
    shape = [int(element) for element in shape]
    return shape

#将预测图片转为模型接收的形状，模型接收的形状用参数shape指定：[H,W,C]
def image_resize(image_path:str,shape:list=None)->np.array:
    image = Image.open(image_path)
    logger.info("图片的格式:{}     尺寸:{}     模式:{}".format(image.format,{"width":image.size[0],"height":image.size[1]},image.mode))
    # 如果模型处理的是单通道图，那么将其转为灰度图
    if shape[2] == 1:
        logger.info("=====转为灰度图=====")
        image = image.convert("L")
    # #Image读取的图片以及相关操作比如下面的resize，都是W,H的顺序
    image = image.resize((shape[1],shape[0]))
    image_numpy = np.array(image)#如果是RGB图则image对象转为numpy，维度顺序为HWC
    image = Image.fromarray(image_numpy)
    image.save("./image_for_predict/t.jpg")
    return image_numpy#如果是灰度图则返回的numpy中不包含通道维度

# image_resize("./image_for_predict/9.jpg",[14,14,1])

def remove_model(dir_number: int):
    try:
        entire_filepath = os.path.join(dir_dict[dir_number], os.listdir(dir_dict[dir_number])[0])
        logger.info("现存的模型:{}".format(entire_filepath))
        os.remove(entire_filepath)
        logger.info("已删除现存的模型")
    except IndexError as e:
        logger.info("{}:模型文件目录为空，不用删除".format(e))
    except Exception as e:
        logger.info("删除出错:{}".format(e))

"""
一般的转换过程是A->onnx->B，前半部分成功率较高，后半部分失败率较高，所以当onnx到B模型失败的话，则将onnx作为目标
模型，复制到目标模型目录下
"""
def copyfiles(source:int,destination:int):
    remove_model(destination)
    sourcefile_name = os.listdir(dir_dict[source])[0]
    entire_source = os.path.join(dir_dict[source],sourcefile_name)
    entire_destination = os.path.join(dir_dict[destination],sourcefile_name)
    shutil.copyfile(entire_source,entire_destination)
    logger.info("已将{}复制到{}".format(entire_source,entire_destination))

def remove_temp_savedmodel():
    shutil.rmtree("./temp_savedmodel")
    os.mkdir("./temp_savedmodel")
