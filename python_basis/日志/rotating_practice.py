#日志级别等级CRITICAL > ERROR > WARNING > INFO > DEBUG。默认为WARNING，可以显示设定level，级别比level小的日志不会产生。

import logging
from logging.handlers import RotatingFileHandler
#给logger取一个名字，这样其他模块中使用getLogger可以获取这个logger。当然也可以直接导入这个logger对象。
logger = logging.getLogger("mingzhu")
logger.setLevel(level=logging.INFO)#比level级别小的日志不会产生
formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s %(funcName)s line=%(lineno)d %(message)s',datefmt="%y-%m-%d %H:%M:%S")

# 定义一个RotatingFileHandler，最多备份10个日志文件(超过10个就把最老的删掉，添加新的)，每个日志文件最大1M
file_handler = RotatingFileHandler("log.txt", maxBytes=1*1024 , backupCount=10)
file_handler.setLevel(logging.INFO)#比该level小的日志不会写到文件里
file_handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)#比该级别小的日志不会打印到控制台
console.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console)

def output_info():
    logger.info("first log information")
    logger.debug("second log information")
    logger.warning("third log information")
    logger.info("fourth log information")
    logger.info("fifth log information")
    logger.info("sixth log information")

logger.info("logger本身设置level相当于限制产生与否;handler设置level相当于过滤")