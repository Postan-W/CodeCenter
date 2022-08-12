"""
当我们需要一个只保存数字的数据结构时，数组比列表更高效，因为其保存的是数字的元素表示(也就是字节)，
而不是对象(Python中即便是数字也是对象),并且数组支持所有跟可变序列有关的操作
"""
from array import array
from random import random
import os
#创建数组需要指定类型码，比如b表示有符号字符(范围从-128到127),d表示双精度浮点数,h表示有符号整数
floats = array('d',(random() for i in range(10**7)))
print(floats[-1])
#将数组保存到文件
if not os.path.exists('floats.bin'):
    with open('floats.bin','wb') as f:
        floats.tofile(f)

floats2 = array('d')
with open('floats.bin','rb') as f:
    floats2.fromfile(f,10**7)#10**7代表读取个数

print(floats == floats2)

