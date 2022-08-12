from base_config import *
rdd_save_path = "file:///C:/Users/15216/Desktop/项目/deeplearing/spark/files/rdd1"
#自定义分区方法
def Partitioner(key):
    print("生成分区")
    return key % 10

def main(filepath=None):
    rdd = sc.parallelize(range(10),5)#5为5个分区
    # 这里的1无意义，只是为了构造键值对，因为partitionBy是将每个元素的key传入分区号计算函数
    rdd2 = rdd.map(lambda i:(i,1))
    #重分区，10代表10个分区
    rdd3 = rdd2.partitionBy(10,Partitioner).map(lambda x:x[0])
    if filepath:
        rdd3.saveAsTextFile(filepath)#10个分区会生成10个文件

if __name__ == "__main__":
    main(rdd_save_path)