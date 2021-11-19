from wtpy.WtUtilDefs import singleton
from wtpy.wrapper import WtDtServoApi
from wtpy.WtCoreDefs import BarList, TickList, WTSBarStruct, WTSTickStruct

from flask import Flask, session, redirect, request, make_response
from flask_compress  import Compress

import urllib.request
import io
import gzip

import json

def pack_rsp(obj):
    rsp = make_response(json.dumps(obj))
    rsp.headers["content-type"]= "text/json;charset=utf-8"
    return rsp

def parse_data():
    try:
        data = request.get_data()
        json_data = json.loads(data.decode("utf-8"))
        return True,json_data
    except:
        return False, {
            "result": -998,
            "message": "请求数据解析失败"
        }

def get_param(json_data, key:str, type=str, defVal = ""):
    if key not in json_data:
        return defVal
    else:
        return type(json_data[key])

def httpPost(url, datas:dict, encoding='utf-8') -> dict:
    headers = {
        'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)',
        'Accept-encoding': 'gzip'
    }
    data = json.dumps(datas).encode("utf-8")
    request = urllib.request.Request(url, data, headers)
    if True:
        f = urllib.request.urlopen(request)
        ec = f.headers.get('Content-Encoding')
        if ec == 'gzip':
            cd = f.read()
            cs = io.BytesIO(cd)
            f = gzip.GzipFile(fileobj=cs)

        ret = json.loads(f.read().decode(encoding))
        f.close()
        return ret
    else:
        return None

@singleton
class WtDtServo:

    # 构造函数，传入动态库名
    def __init__(self):
        self.__config__ = None
        self.__cfg_commited__ = False
        self.local_api = None
        self.server_inst = None
        self.remote_api = None    

    def __check_config__(self):
        '''
        检查设置项\n
        主要会补充一些默认设置项
        '''
        if self.local_api is None:
            self.local_api = WtDtServoApi()

        if self.__config__ is None:
            self.__config__ = dict()

        if "basefiles" not in self.__config__:
            self.__config__["basefiles"] = dict()

        if "data" not in self.__config__:
            self.__config__["data"] = {
                "store":{
                    "path":"./storage/"
                }
            }

    def setRemoteUrl(self, url:str="http://127.0.0.1:8081"):
        if self.__config__ is not None:
            raise Exception('WtDtServo is already in local mode')
            return
        
        self.remote_api = WtDtRemoteServo(url)


    def setBasefiles(self, commfile:str="./common/commodities.json", contractfile:str="./common/contracts.json", 
                holidayfile:str="./common/holidays.json", sessionfile:str="./common/sessions.json", hotfile:str="./common/hots.json"):
        '''
        C接口初始化
        '''
        self.__check_config__()

        self.__config__["basefiles"]["commodity"] = commfile
        self.__config__["basefiles"]["contract"] = contractfile
        self.__config__["basefiles"]["holiday"] = holidayfile
        self.__config__["basefiles"]["session"] = sessionfile
        self.__config__["basefiles"]["hot"] = hotfile

    def setStorage(self, path:str = "./storage/"):
        self.__config__["data"]["store"]["path"] = path
    
    def commitConfig(self):
        if self.remote_api is not None:
            raise Exception('WtDtServo is already in remote mode')
            return
            
        if self.__cfg_commited__:
            return

        cfgfile = json.dumps(self.__config__, indent=4, sort_keys=True)
        try:
            self.local_api.initialize(cfgfile, False)
            self.__cfg_commited__ = True
        except OSError as oe:
            print(oe)

    def __server_impl__(self, port:int, host:str):
        self.server_inst.run(port = port, host = host)
        
    def runServer(self, port:int = 8081, host="0.0.0.0", bSync:bool = True):
        if self.remote_api is not None:
            raise Exception('WtDtServo is already in remote mode')
            return

        app = Flask(__name__)
        app.secret_key = "!@#$%^&*()"
        Compress(app)

        self.server_inst = app

        @app.route("/getbars", methods=["POST"])
        def on_get_bars():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            stdCode = get_param(json_data, "code")
            period = get_param(json_data, "period")
            fromTime = get_param(json_data, "stime", int, None)
            dataCount = get_param(json_data, "count", int, None)
            endTime = get_param(json_data, "etime", int)

            if (fromTime is None and dataCount is None) or (fromTime is not None and dataCount is not None):
                ret = {
                    "result":-1,
                    "message":"Only one of stime and count must be valid at the same time"
                }
            else:
                bars = self.local_api.get_bars(stdCode=stdCode, period=period, fromTime=fromTime, dataCount=dataCount, endTime=endTime)
                if bars is None:
                    ret = {
                        "result":-2,
                        "message":"Data not found"
                    }
                else:
                    bar_list = [curBar.to_dict  for curBar in bars]
                    
                    ret = {
                        "result":0,
                        "message":"Ok",
                        "bars": bar_list
                    }

            return pack_rsp(ret)

        @app.route("/getticks", methods=["POST"])
        def on_get_ticks():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            stdCode = get_param(json_data, "code")
            fromTime = get_param(json_data, "stime", int, None)
            dataCount = get_param(json_data, "count", int, None)
            endTime = get_param(json_data, "etime", int)

            if (fromTime is None and dataCount is None) or (fromTime is not None and dataCount is not None):
                ret = {
                    "result":-1,
                    "message":"Only one of stime and count must be valid at the same time"
                }
            else:
                ticks = self.local_api.get_ticks(stdCode=stdCode, fromTime=fromTime, dataCount=dataCount, endTime=endTime)
                if ticks is None:
                    ret = {
                        "result":-2,
                        "message":"Data not found"
                    }
                else:
                    tick_list = list()
                    for curTick in ticks:
                        curTick = curTick.to_dict
                        curTick["exchg"] = curTick["exchg"].decode()
                        curTick["code"] = curTick["code"].decode()

                        # TODO bid/ask还有问题，先剔除
                        curTick.pop("bid_prices")
                        curTick.pop("ask_prices")
                        curTick.pop("bid_qty")
                        curTick.pop("ask_qty")

                        tick_list.append(curTick)
                    
                    ret = {
                        "result":0,
                        "message":"Ok",
                        "ticks": tick_list
                    }

            return pack_rsp(ret)

        self.commitConfig()
        if bSync:
            self.__server_impl__(port, host)
        else:
            import threading
            self.worker = threading.Thread(target=self.__server_impl__, args=(port,host,))
            self.worker.setDaemon(True)
            self.worker.start()

    def get_bars(self, stdCode:str, period:str, fromTime:int = None, dataCount:int = None, endTime:int = 0) -> BarList:
        '''
        获取K线数据\n
        @stdCode    标准合约代码\n
        @period     基础K线周期，m1/m5/d\n
        @fromTime   开始时间，日线数据格式yyyymmdd，分钟线数据为格式为yyyymmddHHMM\n
        @endTime    结束时间，日线数据格式yyyymmdd，分钟线数据为格式为yyyymmddHHMM，为0则读取到最后一条
        '''
        if self.remote_api is not None:
            return self.remote_api.get_bars(stdCode=stdCode, period=period, fromTime=fromTime, dataCount=dataCount, endTime=endTime)
        
        self.commitConfig()

        if (fromTime is None and dataCount is None) or (fromTime is not None and dataCount is not None):
            raise Exception('Only one of fromTime and dataCount must be valid at the same time')

        return self.local_api.get_bars(stdCode=stdCode, period=period, fromTime=fromTime, dataCount=dataCount, endTime=endTime)

    def get_ticks(self, stdCode:str, fromTime:int = None, dataCount:int = None, endTime:int = 0) -> TickList:
        '''
        获取tick数据\n
        @stdCode    标准合约代码\n
        @fromTime   开始时间，格式为yyyymmddHHMM\n
        @endTime    结束时间，格式为yyyymmddHHMM，为0则读取到最后一条
        '''
        if self.remote_api is not None:
            return self.remote_api.get_ticks(stdCode=stdCode, fromTime=fromTime, dataCount=dataCount, endTime=endTime)

        self.commitConfig()

        if (fromTime is None and dataCount is None) or (fromTime is not None and dataCount is not None):
            raise Exception('Only one of fromTime and dataCount must be valid at the same time')

        return self.local_api.get_ticks(stdCode=stdCode, fromTime=fromTime, dataCount=dataCount, endTime=endTime)

class WtDtRemoteServo:

    def __init__(self, url:str="http://127.0.0.1:8081"):
        self.remote_url = url

    def get_bars(self, stdCode:str, period:str, fromTime:int = None, dataCount:int = None, endTime:int = 0) -> BarList:
        '''
        获取K线数据\n
        @stdCode    标准合约代码\n
        @period     基础K线周期，m1/m5/d\n
        @fromTime   开始时间，日线数据格式yyyymmdd，分钟线数据为格式为yyyymmddHHMM\n
        @endTime    结束时间，日线数据格式yyyymmdd，分钟线数据为格式为yyyymmddHHMM，为0则读取到最后一条
        '''
        if (fromTime is None and dataCount is None) or (fromTime is not None and dataCount is not None):
            raise Exception('Only one of fromTime and dataCount must be valid at the same time')

        url = self.remote_url + "/getbars"
        data = {
            "code":stdCode,
            "period":period,
            "etime":endTime
        }

        if fromTime is not None:
            data["stime"] = fromTime
        elif dataCount is not None:
            data["count"] = dataCount

        resObj = httpPost(url, data)
        if resObj["result"] < 0:
            print(resObj["message"])
            return None

        barCache = BarList()
        for curBar in resObj["bars"]:
            bs = WTSBarStruct()
            bs.date = curBar["date"]
            bs.time = curBar["time"]
            bs.open = curBar["open"]
            bs.high = curBar["high"]
            bs.low = curBar["low"]
            bs.close = curBar["close"]
            bs.settle = curBar["settle"]
            bs.money = curBar["money"]
            bs.vol = curBar["vol"]
            bs.hold = curBar["hold"]
            bs.diff = curBar["diff"]
            barCache.append(bs)
        return barCache
            
        

    def get_ticks(self, stdCode:str, fromTime:int = None, dataCount:int = None, endTime:int = 0) -> TickList:
        '''
        获取tick数据\n
        @stdCode    标准合约代码\n
        @fromTime   开始时间，格式为yyyymmddHHMM\n
        @endTime    结束时间，格式为yyyymmddHHMM，为0则读取到最后一条
        '''
        if (fromTime is None and dataCount is None) or (fromTime is not None and dataCount is not None):
            raise Exception('Only one of fromTime and dataCount must be valid at the same time')

        url = self.remote_url + "/getticks"
        data = {
            "code":stdCode,
            "etime":endTime
        }

        if fromTime is not None:
            data["stime"] = fromTime
        elif dataCount is not None:
            data["count"] = dataCount

        resObj = httpPost(url, data)
        if resObj["result"] < 0:
            print(resObj["message"])
            return None

        tickCache = TickList()
        for curTick in resObj["ticks"]:
            ts = WTSTickStruct()
            ts.exchg = curTick["exchg"].encode('utf-8')
            ts.code = stdCode.encode('utf-8')
            ts.open = curTick["open"]
            ts.high = curTick["high"]
            ts.low = curTick["low"]
            ts.price = curTick["price"]
            ts.settle_price = curTick["settle_price"]

            ts.upper_limit = curTick["upper_limit"]
            ts.lower_limit = curTick["lower_limit"]

            ts.total_volume = curTick["total_volume"]
            ts.volume = curTick["volume"]
            ts.total_turnover = curTick["total_turnover"]
            ts.turn_over = curTick["turn_over"]
            ts.open_interest = curTick["open_interest"]
            ts.diff_interest = curTick["diff_interest"]

            ts.trading_date = curTick["trading_date"]
            ts.action_date = curTick["action_date"]
            ts.action_time = curTick["action_time"]

            ts.pre_close = curTick["pre_close"]
            ts.pre_settle = curTick["pre_settle"]
            ts.pre_interest = curTick["pre_interest"]

            # TODO 还有bid和ask档位没处理

            tickCache.append(ts)
        return tickCache
