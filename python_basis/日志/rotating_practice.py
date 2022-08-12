"""
日志回滚的意思是：
假设日志文件是log2.txt，maxBytes设置的是1*1024，就是1M.当该文件到达1M大小时，会把该文件以名称log2.txt.1保存,
然后新建一个log2.txt文件用来保存接下来的日志信息，
以此类推。而backupCount参数就限定了最多有多少个这样的"备份文件"。
如本模块例子所示，当达到10个备份的时候，再写日志，如果log2.txt装不下了，那么就不会往文件里写信息
"""
import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('time=%(asctime)s || filename=%(filename)s || function=%(funcName)s || line=%(lineno)d || information=%(message)s',datefmt="%y-%m-%d %H:%M:%S")

# 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大1M
rHandler = RotatingFileHandler("log.txt", maxBytes=1*1024 , backupCount=10)
rHandler.setLevel(logging.INFO)
rHandler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(rHandler)
logger.addHandler(console)

def output_info():
    logger.info("first log information")
    logger.debug("second log information")
    logger.warning("third log information")
    logger.info("fourth log information")
    logger.info("fifth log information")
    logger.info("sixth log information")