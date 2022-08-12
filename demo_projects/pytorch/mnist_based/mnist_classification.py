"""
1.一个样本如果只输出一个分类结果(比如是一个概率),这种情况只适合二分类，
所以损失函数包含了两部分：(1-y)log(1-prediction) + ylog(prediction)，即真实标签y为0或1两种情形，如果y为0，那么prediction越接近0损失越小，
如果y为1，那么prediction越接近1损失越小
2.一个样本输出多个结果(比如该样本分属各个类别的概率，和为1),这种情况不仅适合二分类，也适合多分类，其含义是该样本
属于该类别的概率，所以如果真实标签是5，那么只需要取下标为类别5对应的那个概率作为计算的对象即可。下面的多分类交叉熵属于这种情况，
真是标签为torch.longTensor的一维数据，每个元素即是样本所属类别
"""
"""
1）自定义一个类，该类继承自nn.Module类，并且一定要实现两个基本的函数：构造函数__init__()、层的逻辑运算函数forward()；

2）在构造函数__init__()中实现层的参数定义；

3）在前向传播forward函数中实现批数据的前向传播逻辑，只要在nn.Module的子类中定义了forward()函数，backward()函数就会被自动实现。
4）一般把网络中具有可学习参数的层放在构造函数__init()中，当然也可以把不具有可学习参数的层放在里面；

5）如果不具有可学习参数的层放在forward中，则在forward()函数中使用nn.functional代替。

6）forward()方法必须要重写，它是实现模型的功能、实现各个层之间逻辑的核心。

"""
import numpy
import numpy as np
from keras.datasets import mnist
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torch.optim as optim

def get_data()->numpy.array:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()# train是(batch,28,28)的numpy数组；label是（batch，1）的一维numpy数组
    print(x_train.shape,y_train.shape)#(60000, 28, 28) (60000,)
    print(x_test.shape,y_test.shape)#(10000, 28, 28) (10000,)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_test[1][3][10])
    x_train /= 255.0
    x_test /= 255.0
    print(x_test[1][3][10])#经验证是原数据除以255后的值
    x_train = x_train.reshape(x_train.shape[0],1,28,28)
    x_test = x_test.reshape(x_test.shape[0],1,28,28)
    return x_train,y_train,x_test,y_test

get_data()
#在nn的卷积层中，多个通道的卷积结果一般对应位置相加作为一个通道的值
class MnistClassification(torch.nn.Module):
    def __init__(self):
        super(MnistClassification, self).__init__()
        self.convolution1 = torch.nn.Conv2d(1,10,kernel_size=5)#输入单通道，输出10通道，卷积核大小(5,5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入10通道，输出20通道，卷积核大小(5,5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.full_connection1 = torch.nn.Linear(320,60)
        self.full_connection2 = torch.nn.Linear(60,10)

    def forward(self,x):
        x = functional.relu(functional.max_pool2d(self.convolution1(x),2))#此时尺寸为(10,12,12)
        x = functional.relu(functional.max_pool2d(self.convolution2_drop(self.convolution2(x)),2))#此时尺寸为(20,4,4)，所以共320个神经元
        x = x.view(-1,320)#展平操作
        x = functional.relu(self.full_connection1(x))
        x = functional.dropout(x,training=self.training)#training=True,训练的时候才dropout
        x = self.full_connection2(x)
        return functional.log_softmax(x)

#上面的模型因为固定了全连接层的大小，所以只能输入28*28大小的数据，下面采用变量的方式定义输入大小从而动态调整全连接层
"""
经过第一次10通道，5x5，stride=1,padding=0的卷积得到(10,(h-5+1),(w-5+1))
经过一次2x2,stride=2的最大池化,得到(10,int((h-5+1)/2),int((w-5+1))/2)),即不足池化步长的将被舍去。记为(10,h1,w1)
经过一次20通道，5x5,stride=1，padding=0的卷积得到(20,(h1-5+1),(w1-5+1))
经过一次2x2,stride=2的最大池化，得到(20,int((h1-4)/2),int((w1-4)/2)),记为(20,h2,w2)
所以全连接层输入20*h2*w2
"""
class MnistClassificationDynamicInput(torch.nn.Module):
    def __init__(self,h,w):
        super(MnistClassificationDynamicInput,self).__init__()
        h1 = int((h-5+1)/2)
        w1 = int((w-5+1)/2)
        h2 = int((h1-4)/2)
        w2 = int((w1-4)/2)
        self.convolution1 = torch.nn.Conv2d(1,10,kernel_size=5)#输入单通道，输出10通道，卷积核大小(5,5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入10通道，输出20通道，卷积核大小(5,5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.k = 20*h2*w2
        print("全连接层的输入大小为:",self.k)
        self.full_connection1 = torch.nn.Linear(self.k,60)
        self.full_connection2 = torch.nn.Linear(60,10)


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

model = MnistClassificationDynamicInput(28, 28)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

from utility.图片获取 import get_images_labels
x_train, y_train, x_test, y_test = get_data()

def train(epoch,batch_size,x_train,y_train):
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
            print("第{}轮第{}次迭代，损失为:{}".format(epoch,k,loss.data))
    if epoch % 10 == 0:
        print("epoch{}保存模型".format(epoch))
        torch.save(model, "mnist_classification_softmax_epoch{}.pth".format(epoch))


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


def train_and_test(epochs,batch_size,x_train,y_train,x_test,y_test):
    for i in range(1,epochs+1):
        train(i,batch_size,x_train,y_train)
        test(i,x_test,y_test)

# train_and_test(10,100,x_train,y_train,x_test,y_test)


