import numpy as np
import talib as ta


class REPROCESS():
    @staticmethod
    def n() -> int:  # 定义至少需要多少条数据才能计算
        return 0

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:  # 计算方法
        return data


class ZSCORE(REPROCESS):
    @staticmethod
    def n() -> int:
        return 1200

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:
        return (data-ta.MA(data, __class__.n()))/(ta.STDDEV(data, __class__.n())+1e-5)


class ZFILTER(REPROCESS):
    '''
    https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
    '''
    @staticmethod
    def n() -> int:
        return 1200

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:
        # 1e-8
        return (data-ta.MA(data, __class__.n()))/(ta.STDDEV(data, __class__.n())+1e-5)


class MAXMIN(REPROCESS):
    @staticmethod
    def n() -> int:
        return 1200

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:
        data = data[-__class__.n():]
        return (data-data.min())/(data.max()-data.min()+1e-5)*2-1
