"""
@Time : 2021/10/27 10:51
@Author : wmingzhu
@Annotation : 
"""
import torch
from sklearn.datasets import load_boston
import numpy as np
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self,ndim):
        super().__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.randn(ndim,1))#优化器优化的对象是nn.Parameter
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self,x):
        return x.mm(self.weight) + self.bias

boston = load_boston()
linear_model = LinearModel(13)
cretirion = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(),lr=0.000001)
data = torch.tensor(boston["data"],requires_grad=True,dtype=torch.float32)
target = torch.tensor(boston["target"],dtype=torch.float32)

for step in range(10000):
    predict = linear_model(data)
    loss = cretirion(predict,target)
    if step and step % 1000 == 0:
        print("loss:%.3f"%(loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()













