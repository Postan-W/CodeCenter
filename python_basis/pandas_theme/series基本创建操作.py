"""
@Time : 2021/5/11 15:25
@Author : wmingzhu
@Annotation : 
"""
#pandas.Series( data, index, dtype, name, copy),相当于一列数据。index指的是行索引
import pandas as pd
a = [1, 2, 3]
s1 = pd.Series(a)
print(s1)

s2 = pd.Series(a,index=['x','y','z'])
print(s2)
print(s2["x"])

#series类似于字典。也可以用字典来创建
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
s3 = pd.Series(sites)
print(s3)

#只取部分数据
s4 = pd.Series(sites,index=[1,2],name="sites")
print(s4)
print(s4.index)
print(s4.name)