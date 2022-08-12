#加载模型所在模块需要能找到网络结构的定义
import torch
import torch.nn.functional as functional
"""
因为卷积和池化层给定后，linear层的神经元个数可以确定下来，所以输入图片大小可以不限定。输入通道数为1(以mnist数据集为素材)
"""
class MnistClassificationDynamicInput(torch.nn.Module):
    def __init__(self,h,w):
        super(MnistClassificationDynamicInput,self).__init__()
        h1 = int((h-5+1)/2)
        w1 = int((w-5+1)/2)
        h2 = int((h1-4)/2)
        w2 = int((w1-4)/2)
        self.convolution1 = torch.nn.Conv2d(3,10,kernel_size=5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.k = 20*h2*w2
        print("全连接层的输入大小为:",self.k)
        self.full_connection1 = torch.nn.Linear(self.k,60)
        self.full_connection2 = torch.nn.Linear(60,5)

    def forward(self,x):
        x = functional.relu(functional.max_pool2d(self.convolution1(x),2))
        x = functional.relu(functional.max_pool2d(self.convolution2_drop(self.convolution2(x)),2))
        x = x.view(-1,self.k)#展平操作
        x = functional.relu(self.full_connection1(x))
        x = functional.dropout(x,training=self.training)#training=True,训练的时候才dropout
        x = self.full_connection2(x)
        return functional.softmax(x)

