import logging
import os

class WtLogger:

    def __init__(self, catName:str='', filename:str="out.log"):
        self.logger = logging.getLogger(catName)
        self.logger.setLevel(logging.DEBUG)

        #创建一个handler，用于写入日志文件
        log_path = os.getcwd()+"/logs/" # 指定文件输出路径，注意logs是个文件夹，一定要加上/，不然会导致输出路径错误，把logs变成文件名的一部分了
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logname = log_path + filename #指定输出的日志文件名
        fh = logging.FileHandler(logname,encoding = 'utf-8',mode='a')  # 指定utf-8格式编码，避免输出的日志文本乱码
        fh.setLevel(logging.INFO)

        #创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, message:str):
        self.logger.info(message)

    def warn(self, message:str):
        self.logger.warn(message)

    def error(self, message:str):
        self.logger.error(message)

    def fatal(self, message:str):
        self.logger.fatal(message)
