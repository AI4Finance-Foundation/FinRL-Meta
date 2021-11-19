from datetime import datetime

class DBHelper:

    def __init__(self):
        pass

    def initDB(self):
        '''
        初始化数据库，主要是建表等工作
        '''
        pass

    def writeBars(self, bars:list, period="day"):
        '''
        将K线存储到数据库中\n
        @bars   K线序列\n
        @period K线周期
        '''
        pass

    def writeFactors(self, factors:dict):
        '''
        将复权因子存储到数据库中\n
        @factors   复权因子
        '''
        pass


class BaseDataHelper:

    def __init__(self):
        self.isAuthed = False
        pass

    def __check__(self):
        if not self.isAuthed:
            raise Exception("This module has not authorized yet!")

    def auth(self, **kwargs):
        '''
        模块认证
        '''
        pass

    def dmpCodeListToFile(self, filename:str, hasIndex:bool=True, hasStock:bool=True):
        '''
        将代码列表导出到文件\n
        @filename   要输出的文件名，json格式\n
        @hasIndex   是否包含指数\n
        @hasStock   是否包含股票\n
        '''
        pass

    def dmpAdjFactorsToFile(self, codes:list, filename:str):
        '''
        将除权因子导出到文件\n
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]\n
        @filename   要输出的文件名，json格式
        '''
        pass

    def dmpBarsToFile(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        '''
        将K线导出到指定的目录下的csv文件，文件名格式如SSE.600000_d.csv\n
        @folder 要输出的文件夹\n
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]\n
        @start_date 开始日期，datetime类型，传None则自动设置为1990-01-01\n
        @end_date   结束日期，datetime类型，传None则自动设置为当前日期\n
        @period K线周期，支持day、min1、min5\n
        '''
        pass

    def dmpAdjFactorsToDB(self, dbHelper:DBHelper, codes:list):
        '''
        将除权因子导出到数据库\n
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]\n
        @dbHelper   数据库辅助模块
        '''
        pass

    def dmpBarsToDB(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        '''
        将K线导出到数据库\n
        @dbHelper 数据库辅助模块\n
        @codes  股票列表，格式如["SSE.600000","SZSE.000001"]\n
        @start_date 开始日期，datetime类型，传None则自动设置为1990-01-01\n
        @end_date   结束日期，datetime类型，传None则自动设置为当前日期\n
        @period K线周期，支持day、min1、min5\n
        '''
        pass