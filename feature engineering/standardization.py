"""
@Time : 2021/11/8 15:07
@Author : wmingzhu
@Annotation : 
"""
"""
标准化不改变数据分布的形状:上述三种归一化行为的特点是对原始特征的取值减去一个常数然后再除以一个常数，前一步相当于平移，显然不改变数据分布的形状，后一步除以一个数会使原始数据的绝对距离发生改变，但相对距离却不变，所以分布的形状也不变
"""
#以正态分布为例
import numpy as np
import matplotlib.pyplot as plt
# 设置显示中文
normal_data = np.random.normal(loc=200,scale=1000,size=20000)#均值、方差、数量
plt.subplot(2,1,1)
plt.hist(normal_data,bins=100)#以频数显示。bins代表分组数。组距由极差除以分组数得到
plt.title("origin")
print(normal_data.mean(),normal_data.std())
#z-score
zs = (normal_data-200)/1000
print(zs.mean(),zs.std())
plt.subplot(2,1,2)
plt.hist(zs,bins=100)
plt.title("transformed")
plt.show()


