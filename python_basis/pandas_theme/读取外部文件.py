"""
@Time : 2021/11/9 14:22
@Author : wmingzhu
@Annotation : 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def one():
    # 读取文件生成dataframe。如果文件数据没有列名，读取的时候可以通过参数指定，names=["column1","columns",,,]
    data = pd.read_csv("version1.csv")
    # 删除一列然后保存,inplace=1表示执行于这个df对象自身，否则的话这个data对象的该列并没有被删除，只是返回了一个删除了该列的对象
    data.drop(columns="activity", inplace=True)
    print(data.columns)
    # 调整各列位置。把最后一列调整到倒数第二列为例
    columns = data.columns.to_list()
    last = columns.pop()
    columns.insert(-1, last)
    data = data.reindex(columns=columns)
    # 保存。index=0表示不把行序作为一列保存进去
    data.to_csv("modified.csv", index=0)

def two():
    data = pd.read_csv("modified.csv")
    holiday = data["holiday"].value_counts()
    holiday = dict(holiday)
    print(holiday.values())

#绘制直方图
def three():
    data = pd.read_csv("modified.csv")
    total_sales = np.array([int(i) for i in list(data["totalsales"])])
    #随便进行一个平移
    total_sales_transform = total_sales - 100000000
    plt.hist(total_sales,50,)
    plt.hist(total_sales_transform,50)
    plt.legend(["origin","translated"])
    plt.xlabel("group")
    plt.ylabel("value")
    plt.title("7 days total sales")
    plt.show()

three()




