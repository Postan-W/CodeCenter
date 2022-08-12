"""
参考链接：https://blog.csdn.net/lw_zhaoritian/article/details/120991647
进程是系统进行资源分配和调度的独立单位；
线程是进程的实体，是CPU调度和分派的基本单位；
协程也是线程，称微线程，自带CPU上下文，是比线程更小的执行单元;
协程：是一种用户态的轻量级线程，协程的调度完全由用户控制。协程拥有自己的寄存器上下文和栈。 协程调度切换时，将寄存器上下文和栈保存到其他地方，
在切回来的时候，恢复先前保存的寄存器上下文和栈，直接操作栈则基本没有内核切换的开销，可以不加锁的访问全局变量，所以上下文的切换非常快。
"""
import asyncio
import time
import threading

async def test1():
   count = 0
   while True:
       time.sleep(1)
       count += 1
       print("来自test1的当前count是:{}".format(count))
       await asyncio.sleep(8)#sleep阶段，即便对方await了也不会执行，而是对方执行，除非对方await时间更长


async def test2():
    count = 0
    while True:
        time.sleep(3)
        count += 1
        print("来自test2的当前count是:{}".format(count))
        await asyncio.sleep(1)

def use_test_test2():
    loop = asyncio.get_event_loop()#全局只会同时存在一个loop循环事件，也就是说下面得到的loop2和本loop是同一个对象
    loop2 = asyncio.get_event_loop()
    print("=========")
    print(id(loop), id(loop2))  # 从内存id看或从对象hash值看都可以确定多个loop是同一个对象
    print("=========")
    tasks = [test1(),test2()]# 多个协程是并行执行的
    loop.run_until_complete(asyncio.wait(tasks))
    print("=============================================================")
    # tasks2 = [test(1), test2(1), test(2), test2(2), test(3), test2(3)]#6个并行执行
    # loop2.run_until_complete(asyncio.wait(tasks2))
    # loop2.close()
use_test_test2()

#试验一个等待另一个
async def await1():
    print("await1的第一句话,time:{}".format(str(int(time.time()))))
    await asyncio.sleep(2)
    print("await1的第二句话,time:{}".format(str(int(time.time()))))

async def await2():
    print("await2的第一句话,time:{}".format(str(int(time.time()))))
    await await1()
    print("await2的第二句话,time:{}".format(str(int(time.time()))))

def use_await1_await2():
    loop = asyncio.get_event_loop()
    tasks = [await2()]
    loop.run_until_complete(asyncio.wait(tasks))

# use_await1_await2()
#二者并行执行
def use_await1_await2_2():
    loop = asyncio.get_event_loop()
    tasks = [await1(),await2()]
    loop.run_until_complete(asyncio.wait(tasks))

def callback(future):
    print(future.result())
#使用方法创建task
def use_await1_await2_3():
    loop = asyncio.get_event_loop()
    task = loop.create_task(await2())
    #在task执行完毕的时候可以获取执行的结果，回调的最后一个参数是future对象，通过该对象可以获取协程返回值
    # task.add_done_callback(callback)
    loop.run_until_complete(task)

# use_await1_await2_3()



