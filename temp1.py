import asyncio
import queue
import threading
import time

def hello(q:queue.Queue):
    count = 0
    while True:
        time.sleep(4)
        count += 1
        q.put(count)

def task(q:queue.Queue):
    t = threading.Thread(target=hello,args=[q])
    t.setDaemon(True)
    t.start()




