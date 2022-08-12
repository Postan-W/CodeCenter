import queue
import threading
import time
q = queue.Queue(10)

def task_put(q):
    count = 0
    while True:
        time.sleep(1)
        count += 1
        if count >= 30:
            return
        q.put(count,block=1)
        print("放入{}".format(count))

def task_get(q):
    while True:
        time.sleep(3)
        count = q.get(block=1)
        if count >= 29:
            return
        print("获取到:{}".format(count))

def do_task():
    t1 = threading.Thread(target=task_put, args=[q])
    t1.setDaemon(True)
    t1.start()
    t2 = threading.Thread(target=task_get, args=[q])
    t2.setDaemon(True)
    t2.start()


