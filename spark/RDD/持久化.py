"""行动操作触发计算，由于RDD的依赖关系，每次执行一个动作就从头开始沿着依赖关系做转换计算。
可以使用rdd的persist()方法将该rdd标记为持久化，之后执行过一次动作操作后该rdd被持久化了，再次执行别的动作的
时候就不用转换得到这个rdd而是直接使用持久化的内容。
persist的参数:
MEMORY_ONLY:只往内存中持久化。使用该参数时的效果等于rdd直接调用cache()方法
MEMORY_AND_DISK:内存不足时放到磁盘中
"""
from base_config import *

def persistence(hdfs_file=None):
    rdd = sc.parallelize(["Spark","Hadoop","Kafka"])
    rdd.cache()#标记为持久化
    print(rdd.count())#执行了第一个action，持久化生效
    print(",".join(rdd.collect()))#直接利用的是缓存的rdd，而不是从sc.parallelize(["Spark","Hadoop","Kafka"])得到
    rdd.unpersist()#清除缓存

persistence()