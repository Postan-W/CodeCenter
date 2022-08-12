import fastai
from fastai import *
from fastai.vision.all import *
import torch
"""
通过save和load保存和加载模型参数；通过export和load_learner()保存和加载整个模型。
file路径参数若为相对路径，那么则会使用learn.path作为目录,前者只需指定文件名，生成的参数文件格式为pth，放在models下，
后者带上后缀。

"""
# 拉取数据

data = ImageDataLoaders.from_folder(untar_data(URLs.MNIST_SAMPLE),num_workers=0)
learn =cnn_learner(data, models.vgg11_bn, metrics=accuracy)
# 使用learn的fit方法就可以进行训练了，训练一遍
learn.fit(1)
learn.save("./fastaimodel")
# learn.load('./mnist_classi')
