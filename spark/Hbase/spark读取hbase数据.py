"""
将hbase的lib目录下的所有以hbase开头的jar包以及htrace-core-3.1.0-incubating.jar、guava-12.0.1.jar、protobuf-java-2.5.0.jar
这三个jar包(可能spark jars目录下已经存在)复制到spark的jars目录下。
为了让python能够识别读取的hbase数据，还需要spark-examples_2.11-1.6.0-typesafe-001.jar，网上搜索下载后放到spark的jars下
"""
from base_config import *
host = 'localhost'
table = 'student'
conf = {"hbase.zookeeper.quorum":host,"hbase.mapreduce.inputtable":table}
key_converter = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
value_converter = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"
hbase_rdd = sc.newAPIHadoopRDD(
    "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
    "org.apache.hadoop.hbase.io.ImmutableBytesWritable","org.apache.hadoop.hbase.client.Result",keyConverter=key_converter,
    valueConverter=value_converter,conf=conf).cache()


