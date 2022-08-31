import torch
import numpy as np
from PIL import Image
import torch.nn.functional as functional

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

def test1():
    model = torch.load("./mnist_classification_softmax_epoch10.pth")
    trace_module = torch.jit.trace(model, torch.rand(1, 1, 28, 28))
    print(trace_module.code)
    image = Image.open("./6.jpg")
    with_batch = [1]
    with_batch.extend([1, 28, 28])
    # data = np.load("test.npy")
    # data = torch.Tensor(data)
    data = np.array(image).reshape(with_batch)
    prediction = model(data)
    prediction_trace_module = trace_module(data)
    print("torch model prediction:", prediction)
    print("torchscript model prediction:", prediction_trace_module)
    # 保存模型
    torch.jit.save(trace_module,"trace_model.pth")
    # model2 = torch.load("m_c_s_e_tracemodule.pth")
    # prediction3 = model2(data)
    # print(prediction3)

def test2():
    model = MnistClassificationDynamicInput(28,28)
    trace_module = torch.jit.trace(model, torch.rand(1, 1, 28, 28))
    model2 = torch.jit.load("./trace_model.pth")
    image = Image.open("./6.jpg")
    with_batch = [1]
    with_batch.extend([1, 28, 28])
    data = np.array(image).reshape(with_batch)
    # prediction = model2(data)
    # print(prediction)
test2()

"""

"""