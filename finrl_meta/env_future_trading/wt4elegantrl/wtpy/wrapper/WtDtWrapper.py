'''
Descripttion: Automatically generated file comment
version: 
Author: Wesley
Date: 2021-07-27 09:53:43
LastEditors: Wesley
LastEditTime: 2021-08-13 15:26:16
'''
from ctypes import cdll, c_char_p, c_bool, POINTER
from .PlatformHelper import PlatformHelper as ph
from wtpy.WtUtilDefs import singleton
from wtpy.WtCoreDefs import WTSTickStruct, CB_PARSER_EVENT, CB_PARSER_SUBCMD
from wtpy.WtCoreDefs import EVENT_PARSER_CONNECT, EVENT_PARSER_DISCONNECT, EVENT_PARSER_INIT, EVENT_PARSER_RELEASE
import os

# Python对接C接口的库
@singleton
class WtDtWrapper:
    '''
    Wt平台数据组件C接口底层对接模块
    '''

    # api可以作为公共变量
    api = None
    ver = "Unknown"
    
    # 构造函数，传入动态库名
    def __init__(self):
        paths = os.path.split(__file__)
        dllname = ph.getModule("WtDtPorter")
        a = (paths[:-1] + (dllname,))
        _path = os.path.join(*a)
        self.api = cdll.LoadLibrary(_path)
        self.api.get_version.restype = c_char_p
        self.ver = bytes.decode(self.api.get_version())

        self.api.create_ext_parser.restype = c_bool
        self.api.create_ext_parser.argtypes = [c_char_p]

    def run_datakit(self):
        '''
        启动数据组件
        '''
        self.api.start()

    def write_log(self, level, message:str, catName:str = ""):
        '''
        向组件输出日志
        '''
        self.api.write_log(level, bytes(message, encoding = "utf8").decode('utf-8').encode('gbk'), bytes(catName, encoding = "utf8"))

    def initialize(self, cfgfile:str = "dtcfg.json", logprofile:str = "logcfgdt.json"):
        '''
        C接口初始化
        '''
        try:
            self.api.initialize(bytes(cfgfile, encoding = "utf8"), bytes(logprofile, encoding = "utf8"))
        except OSError as oe:
            print(oe)

        self.write_log(102, "WonderTrader datakit initialzied，version：%s" % (self.ver))

    def create_extended_parser(self, id:str) -> bool:
        return self.api.create_ext_parser(bytes(id, encoding = "utf8"))

    def push_quote_from_exetended_parser(self, id:str, newTick:POINTER(WTSTickStruct), bNeedSlice:bool = True):
        return self.api.parser_push_quote(bytes(id, encoding = "utf8"), newTick, bNeedSlice)

    def register_extended_module_callbacks(self,):
        self.cb_parser_event = CB_PARSER_EVENT(self.on_parser_event)
        self.cb_parser_subcmd = CB_PARSER_SUBCMD(self.on_parser_sub)

        self.api.register_parser_callbacks(self.cb_parser_event, self.cb_parser_subcmd)
        self.api.register_exec_callbacks(self.cb_executer_init, self.cb_executer_cmd)

    def on_parser_event(self, evtId:int, id:str):
        id = bytes.decode(id)
        engine = self._engine
        parser = engine.get_extended_parser(id)
        if parser is None:
            return
        
        if evtId == EVENT_PARSER_INIT:
            parser.init(engine)
        elif evtId == EVENT_PARSER_CONNECT:
            parser.connect()
        elif evtId == EVENT_PARSER_DISCONNECT:
            parser.disconnect()
        elif evtId == EVENT_PARSER_RELEASE:
            parser.release()

    def on_parser_sub(self, id:str, fullCode:str, isForSub:bool):
        id = bytes.decode(id)
        engine = self._engine
        parser = engine.get_extended_parser(id)
        if parser is None:
            return
        fullCode = bytes.decode(fullCode)
        if isForSub:
            parser.subscribe(fullCode)
        else:
            parser.unsubscribe(fullCode)
