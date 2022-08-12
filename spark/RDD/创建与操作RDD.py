"""
生成RDD的方式：
1.从文件系统读取文件创建
文件中的一行对应RDD的一个元素,类型为字符串
2.
和通过并行集合(数组)创建。在python中即通过列表创建，列表中的每一个
"""
#行动操作和转换操作模块中有更全面的操作展示
from pyspark import SparkContext,SparkConf
conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)
local_file1 = "file:///C:\\Users\\15216\\Desktop\\hadoop\\iris.txt"
hdfs_file1 = "hdfs://localhost:9000/wmz1/pineapple.txt"
hdfs_file2 = "/wmz1/pineapple.txt"  # 简洁的写法
hdfs_file3 = "hdfs://localhost:9000/wmz/data1.txt"
def use_file(hdfsfile=None):
    # iris_data = sc.textFile(local_file1)
    pineapple = sc.textFile(hdfsfile)  # 想要访问得到相应的Hadoop，需要把Hadoop的配置文件放到spark的conf目录下
    print(pineapple.collect())
    # iris_data.foreach(print)
    pineapple.foreach(print)

# use_file(local_file1)

def use_parallelize():
    rdd = sc.parallelize([1,2,3,4,5,6])
    rdd.foreach(print)
    rdd2 = sc.parallelize([[1,2,3],[4,5,6]])
    rdd2.foreach(print)
# use_parallelize()

def use_filter():
    pineapple = sc.textFile(hdfs_file1)
    new_rdd = pineapple.filter(lambda line:'i' in line)
    new_rdd.foreach(print)

def use_map(hdfs_file):
    rdd = sc.parallelize([1,2,3,4,5])
    new_rdd = rdd.map(lambda x:x+10)
    new_rdd.foreach(print)
    rdd2 = sc.textFile(hdfs_file)
    new_rdd2 = rdd2.map(lambda line:line.split(","))
    new_rdd2.foreach(print)
# use_map(local_file1)
def use_flatmap(hdfsfile):
    rdd = sc.textFile(hdfsfile)
    # flatmap操作相当于先执行map操作，在进行flat操作。以下面为例：相当于先执行(lambda line:line.split(" "))，再
    #执行flat将map中每个可拆分的元素拆为单个RDD元素
    words = rdd.flatMap(lambda line:line.split(","))
    words.foreach(print)
    #假如flatMap操作的是sc.parallelize([1,2,3,4,5])这样的RDD呢
    # array = sc.parallelize([1,2,3,4,5])
    # number = array.flatMap(lambda x:x+10)
    # number.foreach(print)
    #会发现报错。也就说flat的RDD的元素必须是可迭代的
# use_flatmap(local_file1)
def use_groupbykey(hdfs_file=None):
    #groupByKey操作的元素是key-value形式，作用是将相同key的元素合并为一个元素，该元素的key是合并前的，value是所有被合并
    # 的元素的value构成的可迭代对象，具体地在Python中这个对象是ResultIterable Object
    rdd = sc.parallelize([("spark",1),("spark",2),("spark",3),("hadoop",4)])
    print(rdd.collect())
    new_rdd = rdd.groupByKey()
    #('spark', <pyspark.resultiterable.ResultIterable object at 0x000001CBF3812908>)
    #('hadoop', <pyspark.resultiterable.ResultIterable object at 0x00000164AB292908>)
    new_rdd.foreach(lambda keyvalue:print(list(keyvalue[1])))

# use_groupbykey()



