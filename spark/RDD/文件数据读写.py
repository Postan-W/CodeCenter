from base_config import *

def read_and_write_local(local_file=None,save_dir=None,hdfs_save_dir=None):
    rdd = sc.textFile(local_file)#指定三个分区
    print(rdd.first(),type(rdd.first()))
    new_rdd = rdd.flatMap(lambda x:x.split(" ")).map(lambda x:(x,1)).reduceByKey(lambda a,b:a+b).repartition(3)
    print(new_rdd.collect())
    new_rdd.saveAsTextFile(save_dir)
    new_rdd.saveAsTextFile(hdfs_save_dir)
    print("重新读取")
    read_local = sc.textFile(save_dir)
    print(read_local.collect())

# read_and_write_local(local_file2,"file:///C:\\Users\15216\\Desktop\\项目\\PracticeHub\\spark\\files\\rdd7","hdfs://localhost:9000/wmz1/rdd7")


