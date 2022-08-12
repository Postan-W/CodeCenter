from base_config import *

#统计词频
def use_reducebykeys(hdfs_file=None):
    rdd = sc.textFile(hdfs_file)
    #rdd2就是一个键值对rdd。如果map操作结果是可迭代对象，那么flatmap相当于把该对象的元素拆出来作为新RDD的元素
    rdd2 = rdd.flatMap(lambda line:line.split(" ")).map(lambda word:(word,1))
    #统计词频
    rdd3 = rdd2.reduceByKey(lambda a,b:a+b).foreach(print)
    #相当于先执行groupByKey将相同key的元素整合为一个元素，key为原来的key，value为原来的value组成的可迭代对象

#用groupByKey加上map操作实现reduceByKey相同的效果
def groupbykey_map(hdfs_file=None):
    rdd = sc.parallelize(["Spark","Spark","Spark","Hadoop","Hive","Kafka","Python","Python"]).map(lambda word:(word,1))
    rdd2 = rdd.groupByKey()
    rdd3 = rdd2.map(lambda i:(i[0],sum(i[1])))#很好理解。因为value是Python的可迭代对象，所以直接用sum求和
    rdd3.foreach(print)

#通过keys(),values()方法得到包含所有的key或value的RDD。sortByKey()得到根据key的值排序的RDD，默认升序
def get_keys_values():
    rdd = sc.parallelize(["Spark", "Spark", "Spark", "Hadoop", "Hive", "Kafka", "Python", "Python"]).map(
        lambda word: (word, 1))
    rdd_keys = rdd.keys()
    rdd_values = rdd.values()
    rdd_keys.foreach(print)
    rdd_values.foreach(print)
    rdd_sort_asc = rdd.sortByKey()#sortByKey(),默认升序
    rdd_sort_asc.foreach(print)
    rdd_sort_desc = rdd.sortByKey(ascending=False)#sortByKey(ascending=False)，降序
    rdd_sort_desc.foreach(print)
    print(rdd_sort_desc.collect())
    rdd_reduce = rdd.reduceByKey(lambda a,b:a+b).sortBy(lambda x:x[1],False).collect()#sortBy按值降序排列
    print(rdd_reduce)


def use_mapvalues():
    #对rdd的每个元素对的值进行操作，key不变
    rdd = sc.parallelize([("Spark",1),("Hadoop",2),("Hive",3)])
    new_rdd = rdd.mapValues(lambda x:x+1)#传入的是每个元素的值
    print(new_rdd.collect())
# use_mapvalues()
def use_join():
    #join作用是将两个rdd中有相同key的元素合并，key不变，值形成可迭代对象
    rdd1 = sc.parallelize([("Spark",1),("Hadoop",2),("Hive",3)])
    rdd2 = sc.parallelize([("Spark",1),("Spark",2)])
    rdd3 = rdd1.join(rdd2)
    print(rdd3.collect())

use_join()

