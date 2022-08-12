"""
@Time : 2021/11/9 16:25
@Author : wmingzhu
@Annotation : 
"""
import pandas as pd
#行用index表示，列用columns表示
data = pd.read_csv("version1.csv")
print(data.head(3))
print(data.index,data.columns)

#获取前3行数据，先是选取列，然后利用索引选取行
print(data[data.columns][:3])
#单独把一行数据以列表形式取出来
print(list(data.loc[1].values))

#转为numpy数组，数组中的每个元素对应dataframe的一行，所以每个元素的子元素对应列的值
data.values

#使用apply操作某一列的值
data["totalsales"] = data["totalsales"].apply(lambda x:x/10000000)
print(data.head(5))