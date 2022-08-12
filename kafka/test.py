"""
@Time : 2021/11/17 16:58
@Author : wmingzhu
@Annotation : 
"""
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='140.210.92.100:9092')  # 连接kafka

msg = "Hello World".encode('utf-8')  #发送内容,必须是bytes类型
producer.send('wmz', msg)  # 发送的topic为test
producer.close()