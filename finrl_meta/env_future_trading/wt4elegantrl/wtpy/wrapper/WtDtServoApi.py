'''
Descripttion: Automatically generated file comment
version: 
Author: Wesley
Date: 2021-07-27 09:53:43
LastEditors: Wesley
LastEditTime: 2021-08-13 15:34:36
'''
from ctypes import cdll, CFUNCTYPE, c_char_p, c_void_p, c_bool, POINTER, c_uint64, c_uint32
from wtpy.WtCoreDefs import BarList, TickList, WTSBarStruct, WTSTickStruct
from wtpy.wrapper.PlatformHelper import PlatformHelper as ph
from wtpy.WtUtilDefs import singleton

import os

CB_GET_BAR = CFUNCTYPE(c_void_p,  POINTER(WTSBarStruct), c_uint32, c_bool)
CB_GET_TICK = CFUNCTYPE(c_void_p,  POINTER(WTSTickStruct), c_uint32, c_bool)

@singleton
class WtDtServoApi:
    '''
    Wt平台数据组件C接口底层对接模块
    '''

    # api可以作为公共变量
    api = None
    ver = "Unknown"

    # 构造函数，传入动态库名
    def __init__(self):
        paths = os.path.split(__file__)
        dllname = ph.getModule("WtDtServo")
        a = (paths[:-1] + (dllname,))
        _path = os.path.join(*a)
        self.api = cdll.LoadLibrary(_path)

        self.api.get_version.restype = c_char_p
        self.ver = bytes.decode(self.api.get_version())

        self.api.get_bars_by_range.argtypes = [c_char_p, c_char_p, c_uint64, c_uint64, CB_GET_BAR]
        self.api.get_ticks_by_range.argtypes = [c_char_p, c_uint64, c_uint64, CB_GET_TICK]

        self.api.get_bars_by_count.argtypes = [c_char_p, c_char_p, c_uint32, c_uint64, CB_GET_BAR]
        self.api.get_ticks_by_count.argtypes = [c_char_p, c_uint32, c_uint64, CB_GET_TICK]

    def initialize(self, cfgfile:str, isFile:bool):
        self.api.initialize(bytes(cfgfile, encoding = "utf8"), isFile)

    def get_bars(self, stdCode:str, period:str, fromTime:int = None, dataCount:int = None, endTime:int = 0) -> BarList:
        '''
        重采样K线\n
        @stdCode    标准合约代码\n
        @period     基础K线周期，m1/m5/d\n
        @fromTime   开始时间，日线数据格式yyyymmdd，分钟线数据为格式为yyyymmddHHMM\n
        @endTime    结束时间，日线数据格式yyyymmdd，分钟线数据为格式为yyyymmddHHMM，为0则读取到最后一条
        '''
        bar_cache = BarList()
        if fromTime is not None:
            ret = self.api.get_bars_by_range(bytes(stdCode, encoding="utf8"), bytes(period,'utf8'), fromTime, endTime, CB_GET_BAR(bar_cache.on_read_bar))
        else:
            ret = self.api.get_bars_by_count(bytes(stdCode, encoding="utf8"), bytes(period,'utf8'), dataCount, endTime, CB_GET_BAR(bar_cache.on_read_bar))

        if ret == 0:
            return None
        else:
            return bar_cache

    def get_ticks(self, stdCode:str, fromTime:int = None, dataCount:int = None, endTime:int = 0) -> TickList:
        '''
        重采样K线\n
        @stdCode    标准合约代码\n
        @fromTime   开始时间，格式为yyyymmddHHMM\n
        @endTime    结束时间，格式为yyyymmddHHMM，为0则读取到最后一条
        '''
        tick_cache = TickList()
        if fromTime is not None:
            ret = self.api.get_ticks_by_range(bytes(stdCode, encoding="utf8"), fromTime, endTime, CB_GET_TICK(tick_cache.on_read_tick))
        else:
            ret = self.api.get_ticks_by_count(bytes(stdCode, encoding="utf8"), dataCount, endTime, CB_GET_TICK(tick_cache.on_read_tick))

        if ret == 0:
            return None
        else:
            return tick_cache