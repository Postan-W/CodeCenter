#定时任务
import datetime
#定义任务执行的次数
count = 0
#定义一个任务
def task():
    print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
#定时执行任务

#方法1
import time
def method1():
    while True:
        global count
        count += 1
        task()
        time.sleep(2)

#方法2
#APScheduler是一个 Python 定时任务框架，使用起来十分方便。提供了基于日期、固定时间间隔以及 crontab类型的任务，
# 并且可以持久化任务、并以 daemon 方式运行应用。
from apscheduler.schedulers.blocking import BlockingScheduler
def blocking_scheduler():
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    # 参数：要执行的函数，cron是持续百万年的意思，day_of_week是在一周的哪几天，hour和minute是几点几分
    # scheduler.add_job(task, 'cron',hour=16, minute=48)
    # scheduler.add_job(task,'interval',seconds=2)
    scheduler.add_job(task, 'cron',hour=15, minute=54)
    #同一个job下的一个task可能没有运行完，而因为interval比较短，下一个task又该开始了，max_instances就是指定最大并行个数
    # scheduler.add_job(task,'interval',seconds=1,max_instances=1)
    scheduler.start()
blocking_scheduler()

# from apscheduler.schedulers.tornado import TornadoScheduler
# def tornado_scheduler():
#     scheduler = TornadoScheduler(timezone="Asia/Shanghai")
#     scheduler.add_job(task, 'cron',hour=16,minute=44)
#     scheduler.start()
# tornado_scheduler()
