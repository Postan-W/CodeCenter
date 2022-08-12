"""
@Time : 2021/6/1 10:20
@Author : wmingzhu
@Annotation : 
"""
import keras
from utility.model_process import universal_image_process,h5_input_shape
import numpy as np

modelpath = "./simple_cn_model.h5"
imagepath = "./7.jpg"
model = keras.models.load_model(modelpath)
shape = h5_input_shape(model.to_json())
print(shape)
image = universal_image_process(imagepath,shape)
result = model.predict(image)
print(result)
result = np.argmax(result)
print(result)


