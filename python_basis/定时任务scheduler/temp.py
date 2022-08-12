from apscheduler.schedulers.blocking import BlockingScheduler
import datetime
# 输出时间
def job():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

scheduler = BlockingScheduler()
scheduler.add_job(job, 'interval',hours=5,start_date='2022-06-21 16:38:00')
scheduler.start()