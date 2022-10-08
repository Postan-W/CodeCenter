import datetime
import time

#获取只有数值的当前时间串
datetime_clear = lambda :datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ","").replace("-","").replace(":","")
#获取指定格式的日期字符串
str_date = lambda a='%Y-%m-%d %H:%M:%S':datetime.datetime.now().strftime(a)

d1 = datetime.datetime.now()
time.sleep(1)
d2 = datetime.datetime.now()
print(type(d2-d1),d2-d1)#<class 'datetime.timedelta'> 0:00:05.001039
d3 = d2 + datetime.timedelta(days=3)
print(d3)
print(d3-d2)#3 days, 0:00:00