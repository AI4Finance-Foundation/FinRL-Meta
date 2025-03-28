from wtpy.apps.datahelper.DHBaostock import DHBaostock
from wtpy.apps.datahelper.DHDefs import BaseDataHelper
from wtpy.apps.datahelper.DHRqData import DHRqData
from wtpy.apps.datahelper.DHTushare import DHTushare

from meta.data_processors._base import DataSource

class DHFactory:
    @staticmethod
    def createHelper(name: str) -> BaseDataHelper:
        """
        创建数据辅助模块\n
        @name   模块名称，目前支持的有tushare、baostock、rqdata
        """
        name = name.lower()
        if name == DataSource.baostock:
            return DHBaostock()
        elif name == DataSource.tushare:
            return DHTushare()
        elif name == DataSource.rqdata:
            return DHRqData()
        else:
            raise Exception("Cannot recognize helper with name %s" % (name))
