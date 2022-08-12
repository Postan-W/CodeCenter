"""
Date: 2022/04/13
"""
from PIL import Image
import numpy as np
from configparser import ConfigParser
import os
config = ConfigParser()
config.read("./path.ini")
dir_path = config.get("datasets","data2")
files = os.listdir(dir_path)
print("文件夹下共有{}个文件".format(len(files)))

def process_image(dir_path,files):
    processed_dir = os.path.join(dir_path,"processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        for i,file in enumerate(files):
            entire_path = os.path.join(dir_path,file)
            image_format = os.path.splitext(entire_path)[1]
            img = Image.open(entire_path)
            mode = img.mode
            if mode == 'RGB':
                img_resized = img.resize((256, 256))
                save_path = os.path.join(processed_dir, "{}{}".format(i, image_format))
                img_resized.save(save_path)
            else:
                print("===============================NOT RGB============================================")
                print("图片{}".format(entire_path))
                print("的模式是{},所以不作处理".format(mode))
                print("例如RGBA图片可以看到位深度属性为32即8x4即一个像素点占32bit")
                del img
                os.remove(entire_path)
                print("已删除该文件")
                print("===============================NOT RGB============================================")
                continue


process_image(dir_path,files)