from wtpy.apps.datahelper.DHDefs import BaseDataHelper
from wtpy.apps.datahelper.DHBaostock import DHBaostock
from wtpy.apps.datahelper.DHTushare import DHTushare
from wtpy.apps.datahelper.DHRqData import DHRqData

class DHFactory:
    
    @staticmethod
    def createHelper(name:str) -> BaseDataHelper:
        '''
        创建数据辅助模块\n
        @name   模块名称，目前支持的有tushare、baostock、rqdata
        '''
        name = name.lower()
        if name == "baostock":
            return DHBaostock()
        elif name == "tushare":
            return DHTushare()
        elif name == "rqdata":
            return DHRqData()
        else:
            raise Exception("Cannot recognize helper with name %s" % (name))
