#遇到行动操作计算才整正进行。转换操作只是记录了计算的内容和方式
#转换的特点是生成新的RDD，动作的特点的产返回一个结果，比如一个数值或其他数据形式
from base_config import *

def  actions(hdfs_file=None):
    rdd = sc.parallelize([1,2,3,4,5,6,7])
    print(rdd.count())#返回rdd中元素的个数
    print(rdd.first())#返回rdd中的第一个元素
    print(rdd.take(3))#以列表形式返回rdd的前n个元素
    print(rdd.reduce(lambda a,b:a+b))#利用传入的函数对rdd中的元素进行操作返回一个值
    print(rdd.collect())#以列表形式返回rdd的元素
    rdd.foreach(lambda e:print(e))#等同于foreach(print)。取出rdd中的每一个元素给函数使用

actions()

