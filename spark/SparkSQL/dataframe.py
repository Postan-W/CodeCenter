"""
RDD是整个Spark的最基本数据结构，由sparkContext负责将一切数据来源表示为RDD。
相对地，在SparkSQL中(spark2.0开始)由sparkSession负责将一切数据来源转化为DataFrame这种数据结构
RDD转为DF需要构造Row对象，一个Row对象会映射到DF的一行；反之DF转为RDD，也会把一行数据封装到
一个Row对象中，对象的n个属性对应DF的n个列，该Row对象作为RDD的一个元素
dataframe创建来源:文件、pandas、列表
"""
from base_config import *
class DataFrameOperations(ReadAndWrite):
    def __init__(self,df):
        self.df = df

    def print_schema(self):
        print("结构信息：")
        self.df.printSchema()

    def get_selected_dataframe(self,*cols):#自带的select方法也可以传入字符串列表，这里多此一举
        df_columns = []
        for i in range(len(cols)):
            df_columns.append(self.df[cols[i]])
        return self.df.select(*df_columns)

    def filter(self,condition):
        return self.df.filter(condition)

    def get_grouped_dataframe(self,col):
        return self.df.groupBy(col).count()

def read_and_write_test():
    dataframe = read_and_write.fromtext(
        "file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\iris.csv")
    print(dataframe.columns)
    dataframe = spark.createDataFrame(dataframe.toPandas(), schema=["row"])
    """
    上面构造dataframe中用到的schema直接传的是字符串列表，也可以传其原始形式(实际上字符串列表中的每个column都会被构造成原始形式。
    例：
     schema = StructType([StructField("row", StringType(), True)])
     即schema真正使用的是一个StructType对象，其参数为包含n个StructField对象的列表，其中每个StructField对象即是一个列信息描述对象  
    """
    dataframe.show()
    print(type(dataframe))
    dataframe1 = read_and_write.fromtext(
        "file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\iris.csv")
    dataframe1.show(5)
    dataframe2 = read_and_write.fromjson(
        "file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\people.json")
    dataframe2.show(5)
    dataframe3 = dataframe2.drop('age')
    read_and_write.tojson(dataframe3, "file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\peoplename")
# read_and_write_test()

def operations_class():
    path1 = "file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\people.json"
    path2 = "file:///C:\\Users\\15216\Desktop\\projects\\PracticeHub\\spark\\SparkSQL\\people.json"
    operations = DataFrameOperations(read_and_write.fromjson(path2))
    operations.print_schema()
    selected = operations.get_selected_dataframe("name", "age")
    selected.show()
    # 根据年龄降序排序
    operations.df.sort(operations.df["age"].desc()).show()
    # 根据年龄降序，根据姓名升序
    operations.df.sort(operations.df["age"].desc(), operations.df["name"].asc()).show()
# operations_class()
"""
像.txt,.csv这样的文本文件，直接读取为DataFrame时，只会生成一个叫"value"的列，每一行只有这一个列，取值就是文本中的一行数据。可以先把文件读取为RDD，利用RDD丰富的操作对其进行转换，转换过后的RDD来构造pyspark.sql的Row对象，一个Row对
象就作为DataFrame的一行。下面是把文本文件的一行用空格划分词
"""
def rdd_to_dataframe():
    #文件的一行生成一个RDD元素
    rdd = sc.textFile("file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\people.txt")
    print(rdd.collect())
    new_rdd = rdd.map(lambda line:line.split(" ")).map(lambda p:Row(name=p[0],gender=p[1],age=int(p[2])))#构造Row对象
    dataframe = read_and_write.fromtext("file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\people.txt")
    dataframe.show()
    new_dataframe = spark.createDataFrame(new_rdd)
    new_dataframe.show()
    print(new_dataframe.columns)
    read_and_write.tojson(new_dataframe,"file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\peopleinfo")
# rdd_to_dataframe()
#使用sql语句查询DataFrame数据
def dataframe_sql():
    dataframe = read_and_write.fromjson("file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\SparkSQL\\peopleinfo")
    dataframe.show()
    #先将dataframe注册为临时表才能使用SQL语句
    dataframe.createOrReplaceTempView("peopleinfotable")
    #SQL操作的结果以DataFrame返回
    result = spark.sql("select * from peopleinfotable where age > 23")
    result.show()

"""
通过dataframe的rdd属性得到rdd,原dataframe的每一行被封装为Row对象作为rdd的一个元素，
例如Row(age=20, gender='f', name='n1')，列名作为Row对象的属性，所以之后rdd操作里面可以通过属性取值

"""
def dataframe_to_rdd():
    dataframe = read_and_write.fromjson(
        "file:///C:\\Users\\15216\\Desktop\\projects\\PracticeHub\\spark\\SparkSQL\\peopleinfo")
    dataframe.show()
    rdd = dataframe.rdd
    print(rdd.collect())
    new_rdd = rdd.map(lambda p:[p.name,p.gender,p.age])
    print(new_rdd.collect())

# dataframe_to_rdd()


