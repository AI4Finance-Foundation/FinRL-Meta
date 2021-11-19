from wtpy.apps.datahelper.DHDefs import BaseDataHelper, DBHelper
import tushare as ts
from datetime import datetime
import json
import os

def transCode(stdCode:str) -> str:
    items = stdCode.split(".")
    exchg = items[0]
    if exchg == "SSE":
        exchg = "SH"
    elif exchg == "SZSE":
        exchg = "SZ"
    
    if exchg in ['SH','SZ']:
        rawCode = ''
        if len(items) > 2:
            rawCode = items[2]
        else:
            rawCode = items[1]
    else:
        # 期货合约代码，格式为DCE.a.2018
        rawCode = ''
        if exchg == "CZCE":
            rawCode = items[1] + items[2][1:]
        else:
            rawCode = ''.join(items[1:])
    return rawCode.upper() + "." + exchg

    

class DHTushare(BaseDataHelper):

    def __init__(self):
        BaseDataHelper.__init__(self)
        self.api = None
        self.use_pro = True

        print("Tushare helper has been created.")
        return

    def auth(self, **kwargs):
        if self.isAuthed:
            return

        if "use_pro" in kwargs:
            self.use_pro = kwargs["use_pro"]
            kwargs.pop("use_pro")

        self.api = ts.pro_api(**kwargs)
        self.isAuthed = True
        print("Tushare has been authorized, use_pro is %s." % ("enabled" if self.use_pro else "disabled"))

    def dmpCodeListToFile(self, filename:str, hasIndex:bool=True, hasStock:bool=True):
        stocks = {
            "SSE":{},
            "SZSE":{}
        }
        
        #个股列表
        if hasStock:
            print("Fetching stock list...")
            df_stocks = self.api.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            for idx, row in df_stocks.iterrows():
                code = row["ts_code"]
                rawcode = row["symbol"]
                sInfo = dict()
                pid = "STK"
                if code[-2:] == "SH":
                    sInfo["exchg"] = "SSE"
                else:
                    sInfo["exchg"] = "SZSE"
                code = rawcode #code[-2:] + rawcode
                sInfo["code"] = code
                sInfo["name"] = row["name"]
                sInfo["product"] = pid            
                
                stocks[sInfo["exchg"]][code] = sInfo

        if hasIndex:
            #上证指数列表
            print("Fetching index list of SSE...")
            df_stocks = self.api.index_basic(market='SSE')
            for idx, row in df_stocks.iterrows():
                code = row["ts_code"]
                rawcode = code[:6]
                if rawcode[0] != '0':
                    continue
                
                sInfo = dict()
                sInfo["exchg"] = "SSE"
                code = rawcode #"SH" + rawcode
                sInfo["code"] = code
                sInfo["name"] = row["name"]
                sInfo["product"] = "IDX"            
                
                stocks[sInfo["exchg"]][code] = sInfo

            #深证指数列表
            print("Fetching index list of SZSE...")
            df_stocks = self.api.index_basic(market='SZSE')
            for idx, row in df_stocks.iterrows():
                code = row["ts_code"]
                rawcode = code[:6]
                if rawcode[:3] != '399':
                    continue
                
                sInfo = dict()
                sInfo["exchg"] = "SZSE"
                code = rawcode  #"SZ" + rawcode
                sInfo["code"] = code
                sInfo["name"] = row["name"]
                sInfo["product"] = "IDX"            
                
                stocks[sInfo["exchg"]][code] = sInfo

        print("Writing code list into file %s..." % (filename))
        f = open(filename, 'w')
        f.write(json.dumps(stocks, sort_keys=True, indent=4, ensure_ascii=False))
        f.close()


    def dmpAdjFactorsToFile(self, codes:list, filename:str):
        stocks = {
            "SSE":{},
            "SZSE":{}
        }

        count = 0
        length = len(codes)
        for stdCode in codes:
            ts_code = transCode(stdCode)
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            count += 1

            print("Fetching adjust factors of %s(%d/%s)..." % (stdCode, count, length))
            stocks[exchg][code] = list()
            df_factors = self.api.adj_factor(ts_code=ts_code)

            items = list()
            for idx, row in df_factors.iterrows():
                date = row["trade_date"]
                factor = row["adj_factor"]
                items.append({
                    "date": int(date),
                    "factor": float(factor)
                })

            items.reverse()
            pre_factor = 0
            for item in items:
                if item["factor"] != pre_factor:
                    stocks[exchg][code].append(item)
                    pre_factor = item["factor"]

        print("Writing adjust factors into file %s..." % (filename))
        f = open(filename, 'w+')
        f.write(json.dumps(stocks, sort_keys=True, indent=4, ensure_ascii=False))
        f.close()

    def __dmp_bars_to_file_from_pro__(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        freq = ''
        isDay = False
        filetag = ''
        if period == 'day':
            freq = 'D'
            isDay = True
            filetag = 'd'
        elif period == "min5":
            freq = '5min'
            filetag = 'm5'
        elif period == "min1":
            freq = '1min'
            filetag = 'm1'
        else:
            raise Exception("Unrecognized period")

        if isDay:
            start_date = start_date.strftime("%Y%m%d")
            end_date = end_date.strftime("%Y%m%d")
        else:
            start_date = start_date.strftime("%Y-%m-%d") + " 09:00:00"
            end_date = end_date.strftime("%Y-%m-%d") + " 15:15:00"

        count = 0
        length = len(codes)
        for stdCode in codes:
            ts_code = transCode(stdCode)
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            asset_type = "E"
            if (exchg == 'SSE' and code[0] == '0') | (exchg == 'SZSE' and code[:3] == '399'):
                    asset_type =  "I"
            elif exchg not in ['SSE','SZSE']:
                asset_type = "FT"
            count += 1
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, code, count, length))
            df_bars = ts.pro_bar(api=self.api, ts_code=ts_code, start_date=start_date, end_date=end_date, freq=freq, asset=asset_type)
            df_bars = df_bars.iloc[::-1]
            content = "date,time,open,high,low,close,volume,turnover\n"
            for idx, row in df_bars.iterrows():
                if isDay:
                    trade_date = row["trade_date"]
                    date = trade_date + ''
                    time = '0'
                else:
                    trade_time = row["trade_time"]
                    date = trade_time.split(' ')[0]
                    time = trade_time.split(' ')[1]
                o = str(row["open"])
                h = str(row["high"])
                l = str(row["low"])
                c = str(row["close"])
                v = str(row["vol"]*100)
                t = str(row["amount"]*100)
                items = [date, time, o, h, l, c, v, t]

                content += ",".join(items) + "\n"

            filename = "%s.%s_%s.csv" % (exchg, code, filetag)
            filepath = os.path.join(folder, filename)
            print("Writing bars into file %s..." % (filepath))
            f = open(filepath, "w", encoding="utf-8")
            f.write(content)
            f.close()

    def __dmp_bars_to_file_from_old__(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        freq = ''
        isDay = False
        filetag = ''
        if period == 'day':
            freq = 'D'
            isDay = True
            filetag = 'd'
        elif period == "min5":
            freq = '5'
            filetag = 'm5'
        else:
            raise Exception("Unrecognized period")

        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        count = 0
        length = len(codes)
        for stdCode in codes:
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            count += 1
            if (exchg == 'SSE' and code[0] == '0') | (exchg == 'SZSE' and code[:3] == '399'):
                raise Exception("Old api only supports stocks")
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, code, count, length))
            df_bars = ts.get_k_data(code, start=start_date, end=end_date, ktype=freq)
            content = "date,time,open,high,low,close,volume\n"
            for idx, row in df_bars.iterrows():
                if isDay:
                    date = row["date"]
                    time = '0'
                else:
                    trade_time = row["date"]
                    date = trade_time.split(' ')[0]
                    time = trade_time.split(' ')[1] + ":00"
                o = str(row["open"])
                h = str(row["high"])
                l = str(row["low"])
                c = str(row["close"])
                v = str(row["volume"])
                items = [date, time, o, h, l, c, v]

                content += ",".join(items) + "\n"

            filename = "%s.%s_%s.csv" % (exchg, code, filetag)
            filepath = os.path.join(folder, filename)
            print("Writing bars into file %s..." % (filepath))
            f = open(filepath, "w", encoding="utf-8")
            f.write(content)
            f.close()

    def dmpBarsToFile(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if self.use_pro:
            self.__dmp_bars_to_file_from_pro__(folder=folder, codes=codes, start_date=start_date, end_date=end_date, period=period)
        else:
            self.__dmp_bars_to_file_from_old__(folder=folder, codes=codes, start_date=start_date, end_date=end_date, period=period)

    def dmpAdjFactorsToDB(self, dbHelper:DBHelper, codes:list):
        stocks = {
            "SSE":{},
            "SZSE":{}
        }

        count = 0
        length = len(codes)
        for stdCode in codes:
            ts_code = transCode(stdCode)
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            count += 1

            print("Fetching adjust factors of %s(%d/%s)..." % (stdCode, count, length))
            stocks[exchg][code] = list()
            df_factors = self.api.adj_factor(ts_code=ts_code)

            items = list()
            for idx, row in df_factors.iterrows():
                date = row["trade_date"]
                factor = row["adj_factor"]
                items.append({
                    "date": int(date),
                    "factor": factor
                })

            items.reverse()
            pre_factor = 0
            for item in items:
                if item["factor"] != pre_factor:
                    stocks[exchg][code].append(item)
                    pre_factor = item["factor"]

        print("Writing adjust factors into database...")
        dbHelper.writeFactors(stocks)

    def __dmp_bars_to_db_from_pro__(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        freq = ''
        isDay = False
        filetag = ''
        if period == 'day':
            freq = 'D'
            isDay = True
            filetag = 'd'
        elif period == "min5":
            freq = '5min'
            filetag = 'm5'
        elif period == "min1":
            freq = '1min'
            filetag = 'm1'
        else:
            raise Exception("Unrecognized period")

        if isDay:
            start_date = start_date.strftime("%Y%m%d")
            end_date = end_date.strftime("%Y%m%d")
        else:
            start_date = start_date.strftime("%Y-%m-%d") + " 09:00:00"
            end_date = end_date.strftime("%Y-%m-%d") + " 15:15:00"

        count = 0
        length = len(codes)
        for stdCode in codes:
            ts_code = transCode(stdCode)
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            asset_type = "E"
            if (exchg == 'SSE' and code[0] == '0') | (exchg == 'SZSE' and code[:3] == '399'):
                    asset_type =  "I"
            elif exchg not in ['SSE','SZSE']:
                asset_type = "FT"
            count += 1
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, code, count, length))
            df_bars = ts.pro_bar(api=self.api, ts_code=ts_code, start_date=start_date, end_date=end_date, freq=freq, asset=asset_type)
            bars = []
            for idx, row in df_bars.iterrows():          
                if isDay:
                    trade_date = row["trade_date"]
                    bars.append({
                        "exchange":exchg,
                        "code": code,
                        "date": int(trade_date),
                        "time": 0,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["vol"]*100,
                        "turnover": row["amount"]*100
                    })
                else:
                    trade_time = row["trade_time"]
                    date = int(trade_time.split(' ')[0].replace("-",""))
                    time = int(trade_time.split(' ')[1].replace(":","")[:4])
                    bars.append({
                        "exchange":exchg,
                        "code":code,
                        "date": date,
                        "time": time,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["vol"]*100,
                        "turnover": row["amount"]*100
                    })

            print("Writing bars into database...")
            dbHelper.writeBars(bars, period)

    def __dmp_bars_to_db_from_old__(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        freq = ''
        isDay = False
        if period == 'day':
            freq = 'D'
            isDay = True
        elif period == "min5":
            freq = '5'
        else:
            raise Exception("Unrecognized period")

        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        count = 0
        length = len(codes)
        for stdCode in codes:
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            if (exchg == 'SSE' and code[0] == '0') | (exchg == 'SZSE' and code[:3] == '399'):
                raise Exception("Old api only supports stocks")
            count += 1
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, code, count, length))
            df_bars = ts.get_k_data(code, start=start_date, end=end_date, ktype=freq)
            bars = []
            for idx, row in df_bars.iterrows():          
                if isDay:
                    trade_date = row["date"]
                    bars.append({
                        "exchange":exchg,
                        "code": code,
                        "date": int(trade_date.replace('-','')),
                        "time": 0,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"]
                    })
                else:
                    trade_time = row["date"]
                    date = int(trade_time.split(' ')[0].replace("-",""))
                    time = int(trade_time.split(' ')[1].replace(":",""))
                    bars.append({
                        "exchange":exchg,
                        "code":code,
                        "date": date,
                        "time": time,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"]
                    })

            print("Writing bars into database...")
            dbHelper.writeBars(bars, period)

    def dmpBarsToDB(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if self.use_pro:
            self.__dmp_bars_to_db_from_pro__(dbHelper=dbHelper, codes=codes, start_date=start_date, end_date=end_date, period=period)
        else:
            self.__dmp_bars_to_db_from_old__(dbHelper=dbHelper, codes=codes, start_date=start_date, end_date=end_date, period=period)