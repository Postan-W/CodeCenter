"""
@Time : 2021/5/31 11:26
@Author : wmingzhu
@Annotation : 
"""
import os
#比如激活一个conda虚拟环境，然后在该环境下执行操作，那么需要把这两部分在同一个子进程中运行，否则前后两句os.system没有关
print(os.system("conda activate tensorflow-gpu && pip list"))