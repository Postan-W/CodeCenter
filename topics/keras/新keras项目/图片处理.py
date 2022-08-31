from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import to_categorical
import gc
def get_images_labels(limit=0):
    images_dir = r"C:\Users\15216\Desktop\数据集\鲜花\flowers"
    flowers = os.listdir(images_dir)
    flowers_dict = {str(flowers.index(i)): i for i in
                    flowers}  # {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}
    images = []
    labels = []
    print("===========正在读取图片============")
    for subdir in flowers_dict.keys():
        number = 0
        for flower in os.listdir(images_dir + "/" + flowers_dict[subdir]):
            flower_address = images_dir + "/" + flowers_dict[subdir] + "/" + flower
            image = Image.open(flower_address)
            image = image.resize((320, 320))
            images.append(np.array(image))
            labels.append(int(subdir))
            number += 1
            if limit != 0:
                if number >= limit:
                    break
        if limit != 0:
            if number >= limit:
                break

    images = np.array(images).astype("float32") / 255
    print("总的图片个数是:{}".format(len(images)))
    labels = np.array(labels)
    images, labels = shuffle(images, labels)  # 打乱数据顺序
    labels = to_categorical(labels, num_classes=5) #将标签转为one-hot形式
    return images,labels




