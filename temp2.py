import queue
from temp1 import task
import threading
import time

q = queue.Queue()
task(q)

while True:
    time.sleep(5)
    print("获取队列消息:{}".format(q.get()))



