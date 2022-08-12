import numpy
import numpy as np
from keras.datasets import mnist
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torch.optim as optim
from utility.图片获取 import get_images_labels

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
        return functional.softmax(x)#真是标签指定位置的概率值越大，取log后绝对值越小即损失越小，这是合理的

model = MnistClassificationDynamicInput(320, 320)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
images,labels = get_images_labels()
images = images.reshape([images.shape[0],images.shape[3],images.shape[1],images.shape[2]])
labels = np.sum(labels,axis=1).astype("float")
sep = int(len(images)*0.95)
x_train,y_train = images[:sep],labels[:sep]
x_test,y_test = images[sep:],labels[sep:]
def train(batch_size,x_train,y_train):
    iterations = int(x_train.shape[0] / batch_size)
    model.train()#模型设定为训练状态
    print("标签的形状:",y_train.shape)
    for k in range(iterations):
        start,end = k*batch_size,(k+1)*batch_size
        #其中标签构造tensor时需要指定为long类型。torch.from_numpy(y_train).long()直接构造tensor也可以
        data,label = Variable(torch.Tensor(x_train[start:end]),requires_grad=False),Variable(torch.Tensor(y_train[start:end]).type(torch.LongTensor),requires_grad=False)
        optimizer.zero_grad()
        output = model(data)
        """
         torch中交叉熵类的损失函数，标签传的是原始0到n-1的形式，而不是one-hot形式，label维度为(N,)，N为样本数，相当于每个样本的类别索引
        """
        loss = functional.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if k % 200 == 0:
            print("第{}次迭代，损失为:{}".format(k,loss.data))

    torch.save(model, "models/flowers_class.pth")

train(2,x_train,y_train)
#测试
def test(epoch,x_test,y_test):
    model.eval()#设定状态
    data,target = Variable(torch.from_numpy(x_test),volatile=True),Variable(torch.from_numpy(y_test).long())
    """
    预测结果形如[[0.2,0.4,0.4],[0.1,0.6,0.3]]，即每个样本对应一组概率值，一组概率值和为1
    """
    output = model(data)
    test_loss = functional.cross_entropy(output,target).data#取出tensor中的data
    """
    以[[0.2,0.4,0.4],[0.1,0.6,0.3]]作为预测结果为例，output.data.max(1)按照第2维的方向找出最大值，即每组概率的最大值，返回结果形如
    [[0.4,0.6],[1,1]],即最大值列表和其对应的索引，然后[1]即是取出索引，这里即指的是所属类别
    """
    predictions = output.data.max(1)[1]
    #正确个数
    correct = predictions.eq(target.data).cpu().sum()#eq返回形如[False,True,False]的列表，.cpu().sum()使用CPU计算TRUE的个数
    accuracy = correct / len(y_test)
    print("第{}个epoch测试损失:{},正确率:{}".format(epoch,test_loss,accuracy))



