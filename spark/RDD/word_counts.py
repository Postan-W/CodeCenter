
from pyspark import SparkContext
from pyspark import SparkConf
conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)#spark交互式环境中sc是系统自动生成的
words = sc.parallelize(
    ["scala",
     "java",
     "hadoop",
     "spark",
     "akka",
     "spark vs hadoop",
     "pyspark",
     "pyspark and spark"
     ])
counts = words.count()
print(counts)
file = "file:///C:\\Users\\15216\\Desktop\\hadoop\\myStruggle.txt"
text = sc.textFile(file,2).cache()
a_count = text.filter(lambda line:'a' in line).count()
b_count = text.filter(lambda line:'b' in line).count()
print("包含a的行数是{};包含b的行数是{}".format(a_count,b_count))
