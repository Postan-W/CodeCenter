#np.random.rand()本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1

#本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
import numpy as np
print(np.random.rand())
print(np.random.rand(1))
print(np.random.rand(2))
print(np.random.rand(5,2))

#randn()从标准正态分布中返回
print(np.random.randn())
print(np.random.randn(1))
print(np.random.randn(2))
print(np.random.randn(5,2))

#randint(范围,形状)是返回整数
print(np.random.randint(5,10,size=(3,3)))
print(np.random.randint(10,size=(3,3)))#默认从0开始
print("============")
print(np.random.randint(0,5,size=(100)))

#random.random,生成0-1的随机浮点数
print(np.random.random((100,32)))

#shuffle
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
np.random.shuffle(a)
print(a)

#seed函数。每次调用随机函数前，调用seed函数，seed函数的参数值(值的大小无意义)相同则两次随机函数生成的值相同。
print("==================================================================")
np.random.seed(2)
L1 = np.random.randn(3, 3)
np.random.seed(2)
L2 = np.random.randn(3, 3)
np.random.seed(3)
L3 = np.random.randn(3, 3)
print(L1)
print(L2)
print(L3)
