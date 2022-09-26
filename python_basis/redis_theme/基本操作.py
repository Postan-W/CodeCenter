"""
Redis是一个开源的基于内存也可持久化的Key-Value数据库，采用ANSI C语言编写。它拥有丰富的数据结构，拥有事务功能，保证命令的原子性。由于是内存数据库，读写非常高速，可达10w/s的评率，所以一般应用于数据变化快、实时通讯、缓存等。但内存数据库通常要考虑机器的内存大小。
Redis有16个逻辑数据库（db0-db15），每个逻辑数据库项目是隔离的，默认使用db0数据库。若选择第2个数据库，通过命令 select 2 ，python中连接时可以指定数据库。
"""
#关机时，数据默认保存在Redis安装目录dump.rdb文件中
import redis
import numpy as np
redis_connector = redis.Redis(host='127.0.0.1', port= 6379, password= '12345', db= 0)
a1 = redis_connector.get("a1")
print(a1)
