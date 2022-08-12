"""
@Time : 2021/9/8 16:05
@Author : wmingzhu
@Annotation : 
"""
import pandas as pd
import datetime

def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1

csv_path = "C:\\Users\\15216\\Desktop\\POC_20210901_Raw_modified.csv"
target_path = "C:\\Users\\15216\\Desktop\\POC_20210901_Raw_modified.xlsx"
dataframe = pd.read_csv(csv_path)
dataframe = dataframe.dropna(inplace=True)
dataframe.to_csv(csv_path)
# dataframe2 = pd.read_csv(target_path)
#计算流程时间
def get_interval(dataframe):
    pod_arrive = list(dataframe['pod_arrive'])
    pod_remove = list(dataframe['pod_remove'])
    arrive_remove = list(zip(pod_arrive, pod_remove))
    # 年月日时分秒
    arrive_remove_modified = []
    print(arrive_remove[0][0].split(" "))
    print(arrive_remove[0][1].split(" ")[1].split(":"))

    print(type(arrive_remove[0]), arrive_remove[0])

    for element in arrive_remove:
        if type(element[0]) != str or type(element[1]) != str:
            temp1 = []
            temp2 = []
            arrive_remove_modified.append(tuple(zip(temp1, temp2)))
        else:
            data1 = element[0].split(" ")[0].split(".")
            time1 = element[0].split(" ")[1].split(":")
            temp1 = [data1[2], data1[1], data1[0], time1[0], time1[1], time1[2]]
            data2 = element[1].split(" ")[0].split(".")
            time2 = element[1].split(" ")[1].split(":")
            temp2 = [data2[2], data2[1], data2[0], time2[0], time2[1], time2[2]]
            arrive_remove_modified.append((temp1, temp2))

    # 不考虑年和月的差异
    interval = []
    for element in arrive_remove_modified:
        if len(element) == 0:
            interval.append(0)
        else:
            arrive = element[0]
            remove = element[1]
            arrive_date = arrive[0] + "-" + arrive[1] + "-" + arrive[2] + " " + arrive[3] + ":" + arrive[4] + ":" + \
                          arrive[5]
            remove_date = remove[0] + "-" + remove[1] + "-" + remove[2] + " " + remove[3] + ":" + remove[4] + ":" + \
                          remove[5]

            interval_seconds = subtime(arrive_date, remove_date)
            interval.append(interval_seconds)

    dataframe['interval'] = interval


#修正各个日期的格式
def modify_the_date(column:list):
    modified = []
    for i in range(len(column)):
        if type(column[i]) != str:
            column[i] = "01.01.3000 10:10:10"
        date_time = column[i].split(" ")
        date_list = date_time[0].split(".")
        date_str = date_list[2]+"-"+date_list[1]+"-"+date_list[0]
        modified_date_time = date_str + " "+date_time[1]
        modified.append(modified_date_time)
        # print(modified_date_time)
    return modified


def modify_column(dataframe):
    dataframe['pod_arrive'] = modify_the_date(list(dataframe['pod_arrive']))
    dataframe['trackin_time'] = modify_the_date(list(dataframe['trackin_time']))
    dataframe['load_time'] = modify_the_date(list(dataframe['load_time']))
    dataframe['lotstart_time'] = modify_the_date(list(dataframe['lotstart_time']))
    dataframe['lotend_time'] = modify_the_date(list(dataframe['lotend_time']))
    dataframe['trackout_time'] = modify_the_date(list(dataframe['trackout_time']))
    dataframe['unload_time'] = modify_the_date(list(dataframe['unload_time']))
    dataframe['pod_remove'] = modify_the_date(list(dataframe['pod_remove']))
#
# tool_group = dataframe.groupby('tool')
# tools = list(tool_group.groups.keys())
# print(tools)
# for tool in tool_group:
#     print(type(tool[1]))



