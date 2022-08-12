from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession,Row,SQLContext

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
# sc=SparkContext.getOrCreate(conf)
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
sc = spark.sparkContext

local_file1 = "file:///C:\\Users\\15216\\Desktop\\hadoop\\iris.txt"
hdfs_file1 = "hdfs://localhost:9000/wmz1/pineapple.txt"
hdfs_file2 = "/wmz1/pineapple.txt"  # 简洁的写法
hdfs_file3 = "hdfs://localhost:9000/wmz/data1.txt"
local_file2 = "file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\RDD\\data1.txt"



#dataframe的读和写
class ReadAndWrite:
    # 获取文件后缀
    @staticmethod
    def get_format(path):
        return path.split(".")[-1]

    # 读取文件。text、parquet、json。spark.read.text(文件名)这种写法和spark.read.format("text").load(文件名)这种写法都能实现读取
    def fromtext(self,path: str):
        print("读取的文件格式为:{}".format(ReadAndWrite.get_format(path)))
        return spark.read.format("text").load(path)

    def fromjson(self,path: str):
        print("读取的文件格式为:{}".format(ReadAndWrite.get_format(path)))
        return spark.read.format("json").load(path)

    def fromparquet(self,path: str):
        print("读取的文件格式为:{}".format(ReadAndWrite.get_format(path)))
        return spark.read.format("parquet").load(path)

    # 写文件。df.write.text(或json,parquet)与df.write.format(格式).save(路径)效果相同。保存路径为文件夹，而不是一个文件
    def totext(self,df, path: str):
        print("将dataframe保存为:{}".format("text"))
        df.write.format("text").save(path)

    def tojson(self,df, path: str):
        print("将dataframe保存为:{}".format("json"))
        df.write.format("json").save(path)

    def toparquet(self,df, path: str):
        print("将dataframe保存为:{}".format("parquet"))
        df.write.format("parquet").save(path)

read_and_write = ReadAndWrite()
class DataFrameOperations(ReadAndWrite):
    def __init__(self,df):
        self.df = df

    def print_schema(self):
        print("结构信息：")
        self.df.printSchema()

    def get_selected_dataframe(self,*cols):
        df_columns = []
        for i in range(len(cols)):
            df_columns.append(self.df[cols[i]])

        return self.df.select(*df_columns)

    def filter(self,condition):
        return self.df.filter(condition)

    def get_grouped_dataframe(self,col):
        return self.df.groupBy(col).count()