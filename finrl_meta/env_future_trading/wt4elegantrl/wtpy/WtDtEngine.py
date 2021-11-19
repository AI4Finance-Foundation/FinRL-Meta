from wtpy.wrapper import WtDtWrapper
from wtpy.ExtModuleDefs import BaseExtParser
from wtpy.WtUtilDefs import singleton

@singleton
class WtDtEngine:

    def __init__(self):
        self.__wrapper__ = WtDtWrapper()  #api接口转换器
        self.__ext_parsers__ = dict()   #外接的行情接入模块

    def initialize(self, cfgfile:str = "dtcfg.json", logprofile:str = "logcfgdt.json"):
        '''
        数据引擎初始化\n
        @cfgfile    配置文件\n
        @logprofile 日志模块配置文件
        '''
        self.__wrapper__.initialize(cfgfile, logprofile)
    
    def run(self):
        '''
        运行数据引擎
        '''
        self.__wrapper__.run_datakit()

    def add_exetended_parser(self, parser:BaseExtParser):
        '''
        添加扩展parser
        '''
        id = parser.id()
        if id not in self.__ext_parsers__:
            self.__ext_parsers__[id] = parser
            if not self.__wrapper__.create_extended_parser(id):
                self.__ext_parsers__.pop(id)

    def get_extended_parser(self, id:str)->BaseExtParser:
        '''
        根据id获取扩展parser
        '''
        if id not in self.__ext_parsers__:
            return None
        return self.__ext_parsers__[id]

    def push_quote_from_extended_parser(self, id:str, newTick, bNeedSlice:bool):
        '''
        向底层推送tick数据
        '''
        self.__wrapper__.push_quote_from_exetended_parser(id, newTick, bNeedSlice)