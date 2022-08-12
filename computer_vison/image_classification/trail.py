"""
@Time : 2021/6/2 10:45
@Author : wmingzhu
@Annotation : 
"""
import keras
from keras_applications import vgg16
vggmodel = vgg16.VGG16(input_shape=[150,150,3],include_top=False,backend=keras.backend,
                     layers=keras.layers,
                     models=keras.models,
                     utils=keras.utils)
print(type(vggmodel))