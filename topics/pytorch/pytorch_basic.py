import torch
from torch.autograd import Variable
import torch.nn as neuralnetwork
import numpy as np
import matplotlib.pyplot as plt
"""
把网络视为一个函数，在函数中一个变量有其取值也有其导数或者说梯度。torch操作的数据均为tensor，而一个Variable既包含了
tensor即数据也包含了grad等信息
"""
def variable_operation():
    tensor = torch.Tensor([[1,2],[3,4]])#矩阵里的每个值视为一个变量的取值，所以导数矩阵与变量的取值矩阵形状一致
    print(tensor)
    x = Variable(tensor,requires_grad=True)
    print(x)
    y = x +2
    print(y)
    z = y*y*3#z=(x+2)*(x+2)*3。星号乘法是对应位相乘而不是矩阵乘法
    out = z.mean()#z中元素和除以个数
    print(z,out)
    out.backward()#求导
    print(x.grad)
    # print(z.grad)#报错，因为z不是叶子节点张量，backward不会计算它的梯度值

# variable_operation()

#损失函数
def loss_function():
    sample = Variable(torch.ones(2, 2))
    a = torch.Tensor(2, 2)
    a[0, 0] = 0
    a[0, 1] = 1
    a[1, 0] = 2
    a[1, 1] = 3
    target = Variable(a)
    #绝对平均差
    criterion = neuralnetwork.L1Loss()
    loss = criterion(sample,target)
    print(loss)
    #均方误差
    criterion2 = neuralnetwork.MSELoss()
    loss2 = criterion2(sample,target)
    #二分类交叉熵。当然，这里只是功能性测试，这样的数据并不适合二分类交叉熵函数
    criterion3 = neuralnetwork.BCELoss()
    loss3 = criterion3(sample,target)
    print(loss2)
    print(loss3)

#线性回归试验。一元的
def linear_regression():
    x = np.linspace(-1,1,200)
    random_value = np.random.normal(0,0.2,200)
    y = 0.5*x + 0.2*random_value
    plt.scatter(x,y)
    x = Variable(torch.Tensor(x.reshape((200,1))))
    y = Variable(torch.Tensor(y.reshape((200,1))))
    model = torch.nn.Sequential(torch.nn.Linear(1,1))
    optimizer = torch.optim.SGD(model.parameters(),lr=0.5)
    loss_function = torch.nn.MSELoss()
    for i in range(300):
        prediction = model(x)
        loss = loss_function(prediction,y)
        print("均方误差:{}".format(loss))
        """
        利用batch进行梯度下降时(而不是全部训练数据)，如果每次迭代不执行梯度清零操作，相当于上个batch带来的梯度值累加到本batch带来的
        梯度值上面了，有时候或许这样的效果有用。但对于全部数据训练的训练过程，如果把上次的梯度累加到本次，这样的做法看起来不好理解
        ，甚至是荒谬的
        """
        optimizer.zero_grad()
        loss.backward()#反向传播求梯度
        optimizer.step()
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=3)
    plt.show()
    torch.save(model, "models/unitary_linear_regression.pth")

def use_linearregression():
    model = torch.load("models/unitary_linear_regression.pth")
    data = torch.Tensor([[1],[2]])
    print(data)
    predictions_torch = model(data)
    print("------------------------------------------------------------")
    print(predictions_torch)


#模型保存与加载示例
def save_and_load():
    mymodel = None
    # 保存整个神经网络的结构和模型参数
    torch.save(mymodel, 'mymodel.pkl')
    # 只保存神经网络的模型参数
    torch.save(mymodel.state_dict(), 'mymodel_params.pkl')
    mymodel = torch.load('mymodel.pkl')

#Conv2d二维卷积
def conv2d_test():
    a = torch.ones(4,4)
    x = Variable(torch.Tensor(a))
   # 二维卷积的输入：input= (batch,channels,H，W)
    x = x.view(1,1,4,4)#view函数，相当于numpy的resize
    print("输入",x)
    """
    权重，即卷积核weight=(out_channels, in_channels/groups, kH, kW)
    out_channels输出通道数，也反映了卷积核的个数，一个卷积核产生一个通道的数据；in_channels输入通道数
    kH,kW为卷积核高和宽
    """
    weight = Variable(torch.Tensor([[0.1,0.2],[0.3,0.4]])).view(1,1,2,2)
    print("卷积核:",weight)
    y = neuralnetwork.functional.conv2d(input=x,weight=weight,padding=0)#stride默认为1
    print("卷积结果:",y)
# conv2d_test()
"""
卷积核的第一个参数out_channels反映了卷积结果有多少个channel；更重要的是第二个参数in_channels要与输入数据的channel保持
一致，因为运算过程中是对应channel的数据卷积然后求和(或者取平均)
"""
#一维卷积，单通道
def one_channel_conv1d():
    x = Variable(torch.Tensor(range(16))).view(1,1,16)#batch，channel，width
    print(x)
    weight = Variable(torch.Tensor([0.1,0.2,0.3])).view(1,1,3)
    print(weight)
    y = neuralnetwork.functional.conv1d(x,weight,padding=0)#可知y的形状为1,1,14
    print("卷积结果:",y)

# one_channel_conv1d()
#一维卷积，多通道
def multichannel_conv1d():
    x = Variable(torch.Tensor(range(16))).view(1,2,8)
    print(x)#tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],[ 8.,  9., 10., 11., 12., 13., 14., 15.]]])
    weight = Variable(torch.Tensor([0.1,0.2,0.3,1,1,1])).view(1,2,3)
    print(weight)#tensor([[[0.1000, 0.2000, 0.3000],[1.0000, 1.0000, 1.0000]]])
    y = neuralnetwork.functional.conv1d(x,weight,padding=0)#可知y的形状为1,1,6
    print("卷积结果:",y)#tensor([[[27.8000, 31.4000, 35.0000, 38.6000, 42.2000, 45.8000]]])
    """
    以第二个卷积结果为例：(0.1*1+0.2*2+0.3*3)+(1*9+1*10+1*11)=31.4;与Pytorch计算结果一致
    """
# multichannel_conv1d()


#池化
#二维最大池化
def maxpool_2d():
    x = Variable(torch.Tensor(range(20))).view(1,1,4,5)
    print(x)
    y = neuralnetwork.functional.max_pool2d(x,kernel_size=2,stride=2)#kernel_size=2,其中2等同于(2,2)。padding默认为0，不够步长的数据被舍弃
    print("池化结果:",y)
    x_multichannel = Variable(torch.Tensor(range(40))).view(1,2,4,5)
    print(x_multichannel)
    y = neuralnetwork.functional.max_pool2d(x,kernel_size=2,stride=2)
    print(y)#池化不改变通道数，因为是分别对每个channel池化

# maxpool_2d()

#二维平均池化
def averagepool_2d():
    x = Variable(torch.Tensor(range(20))).view(1, 1, 4, 5)
    print(x)
    y = neuralnetwork.functional.avg_pool2d(x, kernel_size=2, stride=2)
    print(y)

# averagepool_2d()

#一维池化
def maxpool_1d():
    x = Variable(torch.Tensor(range(16))).view(1,1,16)
    y = neuralnetwork.functional.max_pool1d(x,kernel_size=2,stride=2)
    print(y)
    x_multichannels = Variable(torch.Tensor(range(16))).view(1,2,8)
    print(x)
    y = neuralnetwork.functional.max_pool1d(x_multichannels, kernel_size=2, stride=2)
    print(y)#tensor([[[ 1.,  3.,  5.,  7.],[ 9., 11., 13., 15.]]])。池化是对每个channel分别池化，不改变channel数
# maxpool_1d()

a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# b = a.data.max(1)[1]
# c = torch.tensor([1,2,2])
# print(b.eq(c.data))
# print(b.eq(c.data).cpu().sum())
# print(a.data.max(0))
b = torch.log(a)#以e为底
print(b)
