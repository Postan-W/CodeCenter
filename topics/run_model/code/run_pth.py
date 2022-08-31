import torch
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as functional
from torch.autograd import Variable
class MnistClassificationDynamicInput(torch.nn.Module):
    def __init__(self,h,w):
        super(MnistClassificationDynamicInput,self).__init__()
        h1 = int((h-5+1)/2)
        w1 = int((w-5+1)/2)
        h2 = int((h1-4)/2)
        w2 = int((w1-4)/2)
        self.convolution1 = torch.nn.Conv2d(3,10,kernel_size=5)#输入单通道，输出10通道，卷积核大小(5,5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入10通道，输出20通道，卷积核大小(5,5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.k = 20*h2*w2
        print("全连接层的输入大小为:",self.k)
        self.full_connection1 = torch.nn.Linear(self.k,60)
        self.full_connection2 = torch.nn.Linear(60,5)

    def forward(self,x):
        x = functional.relu(functional.max_pool2d(self.convolution1(x),2))#此时尺寸为(10,12,12)
        x = functional.relu(functional.max_pool2d(self.convolution2_drop(self.convolution2(x)),2))#此时尺寸为(20,4,4)，所以共320个神经元
        x = x.view(-1,self.k)#展平操作
        x = functional.relu(self.full_connection1(x))
        x = functional.dropout(x,training=self.training)#据说functional的dropout在eval()时不会关闭，nn的会
        x = self.full_connection2(x)
        """
        log_softmax(negative log likelihood loss)相当于把softmax结果取了对数再取负(因为0-1之间的概率值取对数
        结果为负，再加个符号变为正作为这个样本的预测损失值)，那么之后的损失函数就用nll_loss；
        如果这里用的是softmax，那么以后的损失函数就用torch.nn.CrossEntropyLoss(size_average=True)；
        """
        return functional.softmax(x)
model = torch.load("../models/flowers_class.pth")
model.eval()#设定状态
test_images = []
test_image = Image.open("../images/flowers/daisy/5547758_eea9edfd54_n.jpg")
test_image = cv2.imread("../images/flowers/daisy/5547758_eea9edfd54_n.jpg")
test_image = cv2.resize(test_image,(320, 320),interpolation=cv2.INTER_LINEAR)
test_image = test_image[:, :, ::-1].transpose(2, 0, 1)
print(test_image.shape)
test_images.append(test_image)
test_images = np.array(test_images).astype("float32") / 255
shape = test_images.shape
model.eval()
test_images = Variable(torch.from_numpy(test_images),volatile=True)
output = model(test_images)
predictions = output.data.max(1)[1]
print(predictions)
# data,target = Variable(torch.from_numpy(x_test),volatile=True),Variable(torch.from_numpy(y_test).long())
# """
# 预测结果形如[[0.2,0.4,0.4],[0.1,0.6,0.3]]，即每个样本对应一组概率值，一组概率值和为1
# """
# output = model(data)
# test_loss = functional.cross_entropy(output,target).data#取出tensor中的data
# """
# 以[[0.2,0.4,0.4],[0.1,0.6,0.3]]作为预测结果为例，output.data.max(1)按照第2维的方向找出最大值，即每组概率的最大值，返回结果形如
# [[0.4,0.6],[1,1]],即最大值列表和其对应的索引，然后[1]即是取出索引，这里即指的是所属类别
# """
# predictions = output.data.max(1)[1]
# #正确个数
# correct = predictions.eq(target.data).cpu().sum()#eq返回形如[False,True,False]的列表，.cpu().sum()使用CPU计算TRUE的个数
# accuracy = correct / len(y_test)

