"""
Date: 2022/04/26
"""
import os
#执行命令语句
print(os.system("conda info -e > env.txt & conda activate tf2.0 & pip install mxnet==1.7.0.post2 >> env.txt"))
