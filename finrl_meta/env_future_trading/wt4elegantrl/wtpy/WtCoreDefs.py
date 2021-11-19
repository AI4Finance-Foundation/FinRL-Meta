from ctypes import c_uint, c_void_p, CFUNCTYPE, POINTER, c_char_p, c_bool, c_ulong, c_double
from ctypes import Structure, c_char, c_int32, c_uint16, c_uint32, c_uint64, addressof, sizeof
from copy import copy
import numpy as np
import pandas as pd
from typing import Any

MAX_INSTRUMENT_LENGTH = c_char*32
MAX_EXCHANGE_LENGTH = c_char*10
PriceQueueType = c_double*10
VolumeQueueType = c_uint32*10

class WTSStruct(Structure):
    @property
    def fields(self) -> list:
        return self._fields_

    @property
    def values(self) -> tuple:
        return tuple(getattr(self, i[0]) for i in self._fields_)

    @property
    def to_dict(self) -> dict:
        return {i[0]:getattr(self, i[0]) for i in self._fields_}

class WTSTickStruct(WTSStruct):
    '''
    C接口传递的tick数据结构
    '''
    _fields_ = [("exchg", MAX_EXCHANGE_LENGTH),
                ("code", MAX_INSTRUMENT_LENGTH),
                ("price", c_double),
                ("open", c_double),
                ("high", c_double),
                ("low", c_double),
                ("settle_price", c_double),

                ("upper_limit", c_double),
                ("lower_limit", c_double),

                ("total_volume", c_uint32),
                ("volume", c_uint32),
                ("total_turnover", c_double),
                ("turn_over", c_double),
                ("open_interest", c_uint32),
                ("diff_interest", c_int32),

                ("trading_date", c_uint32),
                ("action_date", c_uint32),
                ("action_time", c_uint32),

                ("pre_close", c_double),
                ("pre_settle", c_double),
                ("pre_interest", c_uint32),

                ("bid_prices", PriceQueueType),
                ("ask_prices", PriceQueueType),
                ("bid_qty", VolumeQueueType),
                ("ask_qty", VolumeQueueType)]
    _pack_ = 1

    @property
    def fields(self) -> list:
        fields = self._fields_.copy()
        fields[0] = ('exchg', 'S10')
        fields[1] = ('code', 'S10')
        fields[-4] = ('bid_prices', 'O')
        fields[-3] = ('ask_prices', 'O')
        fields[-2] = ('bid_qty', 'O')
        fields[-1] = ('ask_qty', 'O')
        return fields


class WTSBarStruct(WTSStruct):
    '''
    C接口传递的bar数据结构
    '''
    _fields_ = [("date", c_uint32),
                ("time", c_uint32),
                ("open", c_double),
                ("high", c_double),
                ("low", c_double),
                ("close", c_double),
                ("settle", c_double),
                ("money", c_double),
                ("vol", c_uint32),
                ("hold", c_uint32),
                ("diff", c_int32)]
    _pack_ = 1

class WTSTransStruct(WTSStruct):
    '''
    C接口传递的逐笔成交数据结构
    '''
    _fields_ = [("exchg", MAX_EXCHANGE_LENGTH),
                ("code", MAX_INSTRUMENT_LENGTH),

                ("trading_date", c_uint32),
                ("action_date", c_uint32),
                ("action_time", c_uint32),

                ("index", c_uint32),
                ("ttype", c_int32),
                ("side", c_int32),

                ("price", c_double),
                ("volume", c_uint32),
                ("askorder", c_int32),
                ("bidorder", c_int32)]
    _pack_ = 1

class WTSOrdQueStruct(WTSStruct):
    '''
    C接口传递的委托队列数据结构
    '''
    _fields_ = [("exchg", MAX_EXCHANGE_LENGTH),
                ("code", MAX_INSTRUMENT_LENGTH),

                ("trading_date", c_uint32),
                ("action_date", c_uint32),
                ("action_time", c_uint32),

                ("side", c_int32),
                ("price", c_double),
                ("order_items", c_uint32),
                ("qsize", c_uint32),
                ("volumes", c_uint32*50)]
    _pack_ = 1

class WTSOrdDtlStruct(WTSStruct):
    '''
    C接口传递的委托明细数据结构
    '''
    _fields_ = [("exchg", MAX_EXCHANGE_LENGTH),
                ("code", MAX_INSTRUMENT_LENGTH),

                ("trading_date", c_uint32),
                ("action_date", c_uint32),
                ("action_time", c_uint32),

                ("index", c_uint32),
                ("side", c_int32),
                ("price", c_double),
                ("volume", c_uint32),
                ("otype", c_int32)]
    _pack_ = 1


class CacheList(list):
    def to_record(self) -> np.recarray:
        data = np.empty(len(self), dtype=self[0].fields)
        for k, v in enumerate(self):
            data[k] = v.values
        return data.view(np.recarray)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_record())

class BarList(CacheList):
    def on_read_bar(self, curBar:POINTER(WTSBarStruct), count:int, isLast:bool):
        bsSize = sizeof(WTSBarStruct)
        addr = addressof(curBar.contents)
        for i in range(count):
            thisBar = WTSBarStruct.from_address(addr)
            self.append(copy(thisBar))
            addr += bsSize

    def on_data_count(self, count:int):
        pass

class TickList(CacheList):
    def on_read_tick(self, curTick:POINTER(WTSTickStruct), count:int, isLast:bool):
        tsSize = sizeof(WTSTickStruct)
        addr = addressof(curTick.contents)
        for i in range(count):
            thisTick = WTSTickStruct.from_address(addr)
            self.append(copy(thisTick))
            addr += tsSize

    def on_data_count(self, count:int):
        pass

# 回调函数定义
#策略初始化回调
CB_STRATEGY_INIT = CFUNCTYPE(c_void_p, c_ulong) 
#策略tick数据推送回调
CB_STRATEGY_TICK = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSTickStruct))
#策略获取tick数据的单条tick同步回调
CB_STRATEGY_GET_TICK = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSTickStruct), c_uint32, c_bool)
#策略重算回调(CTA/SEL策略)
CB_STRATEGY_CALC = CFUNCTYPE(c_void_p, c_ulong, c_ulong, c_ulong)
#策略订阅的K线闭合事件回调
CB_STRATEGY_BAR = CFUNCTYPE(c_void_p, c_ulong, c_char_p, c_char_p, POINTER(WTSBarStruct))
#策略获取K线数据的单条K线同步回调
CB_STRATEGY_GET_BAR = CFUNCTYPE(c_void_p, c_ulong, c_char_p, c_char_p, POINTER(WTSBarStruct), c_uint32, c_bool)
#策略获取全部持仓的同步回调
CB_STRATEGY_GET_POSITION = CFUNCTYPE(c_void_p, c_ulong, c_char_p, c_double, c_bool)
#交易日开始结束事件回调
CB_SESSION_EVENT = CFUNCTYPE(c_void_p, c_ulong, c_ulong, c_bool) 

#引擎事件回调(交易日开启结束等)
CB_ENGINE_EVENT = CFUNCTYPE(c_void_p, c_ulong, c_ulong, c_ulong)

#HFT策略交易通道事件回调
CB_HFTSTRA_CHNL_EVT = CFUNCTYPE(c_void_p, c_ulong, c_char_p, c_ulong)
#HFT策略订单推送回报
CB_HFTSTRA_ORD = CFUNCTYPE(c_void_p, c_ulong, c_ulong, c_char_p, c_bool, c_double, c_double, c_double, c_bool, c_char_p)
#HFT策略成交推送回报
CB_HFTSTRA_TRD = CFUNCTYPE(c_void_p, c_ulong, c_ulong, c_char_p, c_bool, c_double, c_double, c_char_p)
#HFT策略下单结果回报
CB_HFTSTRA_ENTRUST = CFUNCTYPE(c_void_p, c_ulong, c_ulong, c_char_p, c_bool, c_char_p, c_char_p)

#策略委托队列推送回调
CB_HFTSTRA_ORDQUE = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSOrdQueStruct))
#策略获取委托队列数据的单条数据同步回调
CB_HFTSTRA_GET_ORDQUE = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSOrdQueStruct), c_uint32, c_bool)
#策略委托明细推送回调
CB_HFTSTRA_ORDDTL = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSOrdDtlStruct))
#策略获取委托明细数据的单条数据同步回调
CB_HFTSTRA_GET_ORDDTL = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSOrdDtlStruct), c_uint32, c_bool)
#策略成交明细推送回调
CB_HFTSTRA_TRANS = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSTransStruct))
#策略获取成交明细数据的单条数据同步回调
CB_HFTSTRA_GET_TRANS = CFUNCTYPE(c_void_p, c_ulong, c_char_p, POINTER(WTSTransStruct), c_uint32, c_bool)


EVENT_ENGINE_INIT	= 1     #框架初始化
EVENT_SESSION_BEGIN = 2     #交易日开始
EVENT_SESSION_END	= 3     #交易日结束
EVENT_ENGINE_SCHDL	= 4     #框架调度
EVENT_BACKTEST_END  = 5     #回测结束

CHNL_EVENT_READY	= 1000  #通道就绪事件
CHNL_EVENT_LOST		= 1001  #通道断开事件

from enum import Enum
class EngineType(Enum):
    '''
    引擎类型
    枚举变量
    '''
    ET_CTA = 999
    ET_HFT = 1000
    ET_SEL = 1001

        
'''
Parser外接实现
'''
EVENT_PARSER_INIT		= 1;	#Parser初始化
EVENT_PARSER_CONNECT	= 2;	#Parser连接
EVENT_PARSER_DISCONNECT = 3;	#Parser断开连接
EVENT_PARSER_RELEASE	= 4;	#Parser释放
CB_PARSER_EVENT = CFUNCTYPE(c_void_p, c_ulong, c_char_p)
CB_PARSER_SUBCMD = CFUNCTYPE(c_void_p, c_char_p, c_char_p, c_bool)

'''
Executer外接实现
'''
CB_EXECUTER_INIT = CFUNCTYPE(c_void_p, c_char_p)
CB_EXECUTER_CMD = CFUNCTYPE(c_void_p, c_char_p, c_char_p, c_double)
