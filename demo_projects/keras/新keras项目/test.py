# from tensorflow.keras.layers import concatenate
# import numpy as np
# a= np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])#2,2,3
# b = np.array([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
# c = concatenate([a,b])
# print(c)
# testdata1 = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[4,4,4,4],[5,5,5,5],[6,6,6,6]]])
# testdata2 = np.array([[[7,7,7,7],[8,8,8,8],[9,9,9,9]],[[10,10,10,10],[11,11,11,11],[12,12,12,12]]])
# con_result = concatenate([testdata1,testdata2])
# print(con_result)
#=================================================================
# from tensorflow.keras.datasets import cifar10
# from PIL import Image
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# image = Image.fromarray(x_train[0])
# image.show()
#=====================================================================
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.datasets import cifar10,mnist
# from utility.图片获取 import get_images_labels_unstandarization
# from tensorflow.keras.utils import to_categorical
# x_train,y_train = get_images_labels_unstandarization()
# y_train = to_categorical(y_train)
# print(y_train)
# print(x_train.shape)
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
#     width_shift_range=0.1,  # randomly shift images horizontally
#     height_shift_range=0.1,  # randomly shift images vertically
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False)  # randomly flip images
# """
# fit不是必须的，当ImageDataGenerator构造函数中需要使用
#
# featurewise_center:
# samplewise_center:
# featurewise_std_normalization:
# samplewise_std_normalization:
# 这几个参数时才需要使用fit方法，因为需要从fit方法中得到原始图形的统计信息，比如均值、方差等等
# """
# datagen.fit(x_train)
# data1 = datagen.flow(x_train, y_train, batch_size=4)
# print(data1[0])
#====================================================================
import numpy as np
from tensorflow.keras.layers import BatchNormalization
a= np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
mean = np.mean(a,axis=1,keepdims=True)
std = np.std(a,axis=-1,keepdims=True)
print(mean)









