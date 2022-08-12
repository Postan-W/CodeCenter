"""
@Time : 2021/11/8 16:36
@Author : wmingzhu
@Annotation : 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#调整下列名
def change_column_name():
    data = pd.read_csv("sales_data.csv")
    columns = list(data.columns)
    columns_modified = [i.split(".")[1] for i in columns]
    data.columns = columns_modified
    data.to_csv("version1.csv",index=0)#不保留行索引

#将销售额的小数去掉
def version1():
    data = pd.read_csv("version1.csv")
    total_sales_float = list(data["totalsales"])
    total_sales_int = [int(str(i).split(".")[0]) for i in total_sales_float]
    data["totalsales"] = total_sales_int
    data.to_csv("version2.csv",index=0)

#特征间的数值差异性较大,标准化处理。处理members相关特征
def version2():
    data = pd.read_csv("version2.csv")
    #members2,members3数据太稀疏，合并到members1中处理
    members1 = np.array(list(data["members1"]))
    members2 = np.array(list(data["members2"]))
    members3 = np.array(list(data["members3"]))
    members_total = members1 + members2 + members3
    # plt.subplot(3,1,1)
    # plt.hist(members_total,bins=100)
    # members_total_mean = int(members_total.mean())
    # members_total_std = int(members_total.std())
    # members_total_zscore = (members_total - members_total_mean)/members_total_std
    # plt.subplot(3,1,2)
    # plt.hist(members_total_zscore,bins=100)
    # print(members_total_zscore[:10])
    # plt.subplot(3,1,3)
    # members_total_max = max(members_total)
    # members_total_min = min(members_total)
    # members_total_minmax_normalization = (members_total -members_total_min)/members_total_max
    # plt.hist(members_total_minmax_normalization,bins=100)
    # print(members_total_minmax_normalization[:3])
    # plt.show()
    data.drop(columns=["members1","members2","members3"],inplace=True)
    data["members"] = members_total
    columns = data.columns.to_list()
    columns.pop()
    columns.insert(-1,"members")
    data = data.reindex(columns=columns)
    data.to_csv("version3.csv",index=0)

#date_time列删掉；activity列删掉，因为该列过于稀疏，并且对目标列几乎没有影响；不作归一化，因为会员数和访问人数都是不定的，无穷的，如果都归一到0-1区间，新数据的100000和训练数据的100可能会有相同的数值，然而他们不应该相同，因为他们对总销售额的影响差异巨大
def version3():
    data = pd.read_csv("version3.csv")
    print(data.columns.to_list())
    data.drop(columns=['date_time', 'activity'],inplace=True)
    data.to_csv("version4.csv",index=0)

#将人数以10万为单位、会员数以万为单位、总销售额以1亿为单位
def version4():
    data = pd.read_csv("version4.csv")
    nop = np.round(np.array(list(data["nop"])) / 100000).astype(int)
    data["nop"] = nop
    members = np.round(np.array(list(data["members"])) / 10000).astype(int)
    data["members"] = members
    totalsales = np.round(np.array(list(data["totalsales"])) / 100000000).astype(int)
    data["totalsales"] = totalsales
    data.to_csv("version5.csv",index=0)

version4()
















