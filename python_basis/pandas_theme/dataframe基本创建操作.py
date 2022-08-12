"""
@Time : 2021/5/11 15:18
@Author : wmingzhu
@Annotation : 
"""
"""
DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
pandas.DataFrame( data, index, columns, dtype, copy)

"""
import pandas as pd

dict1 = {"name":["Elizabeth","Michael","Claude"],"gender":["female","male","male"]}

dataframe1 = pd.DataFrame(dict1)

print(dataframe1)

#使用列表创建，每个子元素相当于一行
list1 = [['Google',10],['Runoob',12],['Wiki',13]]
df2 = pd.DataFrame(list1,columns=['Site','Age'],dtype=float)
print(df2)
#与上面列表创建等效的是,每个key相当于列名
list2 = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df3 = pd.DataFrame(list2)
print (df3)

#使用loc返回行数据
print(dataframe1.loc[0])

#指定行索引
df4 = pd.DataFrame(dict1,index=['user1','user2','user3'])
print(df4)
print(df4.loc["user1"])