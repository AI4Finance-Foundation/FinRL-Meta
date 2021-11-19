from wtpy.apps.datahelper.DHDefs import BaseDataHelper, DBHelper
import rqdatac as rq
from datetime import datetime, timedelta
import json
import os

def exchgStdToRQ(exchg:str) -> str:
    if exchg == 'SSE':
        return "XSHG"
    elif exchg == 'SZSE':
        return "XSHE"
    else:
        return exchg

def exchgRQToStd(exchg:str) -> str:
    if exchg == 'XSHG':
        return "SSE"
    elif exchg == 'XSHE':
        return "SZSE"
    else:
        return exchg

def stdCodeToRQ(stdCode:str):
    stdCode = stdCode.upper()
    items = stdCode.split(".")
    exchg = exchgStdToRQ(items[0])
    if len(items) == 2:
        # 简单股票代码，格式如SSE.600000
        return items[1] + "." + exchg
    elif items[1] in ["IDX","ETF","STK","OPT"]:
        # 标准股票代码，格式如SSE.IDX.000001
        return items[2] + "." + exchg
    elif len(items) == 3:
        # 标准期货代码，格式如CFFEX.IF.2103
        if items[2] != 'HOT':
            return ''.join(items[1:])
        else:
            return items[1] + "88"

    


class DHRqData(BaseDataHelper):

    def __init__(self):
        BaseDataHelper.__init__(self)
        print("Rqdata helper has been created.")
        return

    def auth(self, **kwargs):
        if self.isAuthed:
            return

        rq.init(**kwargs)
        self.isAuthed = True
        print("Rqdata has been authorized.")

    def dmpCodeListToFile(self, filename:str, hasIndex:bool=True, hasStock:bool=True):
        stocks = {
            "SSE":{},
            "SZSE":{}
        }
        
        #个股列表
        if hasStock:
            print("Fetching stock list...")
            df_stocks = rq.all_instruments(type='CS', market="cn")
            for idx, row in df_stocks.iterrows():
                rawcode = row["order_book_id"][:6]
                exchg = row["exchange"]
                if exchg == 'XSHG':
                    exchg = "SSE"
                else:
                    exchg = "SZSE"
                sInfo = dict()
                sInfo["exchg"] = exchg                    
                sInfo["code"] = rawcode
                sInfo["name"] = row["symbol"]
                sInfo["product"] = "STK"            
                
                stocks[sInfo["exchg"]][rawcode] = sInfo

        if hasIndex:
            #上证指数列表
            print("Fetching index list...")
            df_stocks = rq.all_instruments(type='INDX', market="cn")
            for idx, row in df_stocks.iterrows():
                rawcode = row["order_book_id"][:6]
                exchg = row["exchange"]
                if exchg == 'XSHG':
                    exchg = "SSE"
                else:
                    exchg = "SZSE"
                sInfo = dict()
                sInfo["exchg"] = exchg                    
                sInfo["code"] = rawcode
                sInfo["name"] = row["symbol"]
                sInfo["product"] = "IDX"            
                
                stocks[sInfo["exchg"]][rawcode] = sInfo

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
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            count += 1
            rq_code = code + "." + exchgStdToRQ(exchg)

            stocks[exchg][code] = list()
            print("Fetching adjust factors of %s(%d/%s)..." % (code, count, length))
            df_factors = rq.get_ex_factor(order_book_ids=rq_code, start_date="1990-01-01")
    
            for idx, row in df_factors.iterrows():
                date = row['announcement_date'].to_pydatetime()
                date = date + timedelta(days=1)
                factor = float(row['ex_cum_factor'])
                stocks[exchg][code].append({
                    "date": int(date.strftime("%Y%m%d")),
                    "factor": factor
                })
        
        print("Writing adjust factors into file %s..." % (filename))
        f = open(filename, 'w+')
        f.write(json.dumps(stocks, sort_keys=True, indent=4, ensure_ascii=False))
        f.close()

    def dmpBarsToFile(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        freq = ''
        isDay = False
        filetag = ''
        if period == 'day':
            freq = '1d'
            isDay = True
            filetag = 'd'
        elif period == "min5":
            freq = '5m'
            filetag = 'm5'
        elif period == "min1":
            freq = '1m'
            filetag = 'm1'
        else:
            raise Exception("Unrecognized period")
        
        count = 0
        length = len(codes)
        for stdCode in codes:
            count += 1
            rq_code = stdCodeToRQ(stdCode)
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, stdCode, count, length))
            df_bars = rq.get_price(order_book_ids = rq_code,start_date=start_date, end_date=end_date,frequency=freq,adjust_type='none',expect_df=True)
            content = "date,time,open,high,low,close,volume,turnover,hold\n"
            total_nums = len(df_bars)
            cur_num = 0
            for idx, row in df_bars.iterrows():
                trade_date = row.name[1].to_pydatetime()
                date = trade_date.strftime("%Y-%m-%d")
                if isDay:
                    time = '0'
                else:
                    time = trade_date.strftime("%H:%M:%S")
                o = str(row["open"])
                h = str(row["high"])
                l = str(row["low"])
                c = str(row["close"])
                v = str(row["volume"])
                t = str(row["total_turnover"])
                items = [date, time, o, h, l, c, v, t]
                if "open_interest" in row:
                    items.append(str(row["open_interest"]))

                content += ",".join(items) + "\n"

                cur_num += 1
                if cur_num % 500 == 0:
                    print("Processing bars %d/%d..." % (cur_num, total_nums))

            filename = "%s_%s.csv" % (stdCode, filetag)
            filepath = os.path.join(folder, filename)
            print("Writing bars into file %s..." % (filepath))
            f = open(filepath, "w", encoding="utf-8")
            f.write(content)
            f.close()

    def dmpAdjFactorsToDB(self, dbHelper:DBHelper, codes:list):
        stocks = {
            "SSE":{},
            "SZSE":{}
        }

        count = 0
        length = len(codes)
        for stdCode in codes:
            exchg = stdCode.split(".")[0]
            code = stdCode[-6:]
            count += 1
            rq_code = code + "." + exchgStdToRQ(exchg)

            stocks[exchg][code] = list()
            print("Fetching adjust factors of %s(%d/%s)..." % (code, count, length))
            df_factors = rq.get_ex_factor(order_book_ids=rq_code, start_date="1990-01-01")
    
            for idx, row in df_factors.iterrows():
                date = row['announcement_date'].to_pydatetime()
                date = date + timedelta(days=1)
                factor = float(row['ex_cum_factor'])
                stocks[exchg][code].append({
                    "date": int(date.strftime("%Y%m%d")),
                    "factor": factor
                })
        
        print("Writing adjust factors into database...")
        dbHelper.writeFactors(stocks)

    def dmpBarsToDB(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        freq = ''
        isDay = False
        if period == 'day':
            freq = '1d'
            isDay = True
        elif period == "min5":
            freq = '5m'
        elif period == "min1":
            freq = '1m'
        else:
            raise Exception("Unrecognized period")
        
        count = 0
        length = len(codes)
        for stdCode in codes:
            items = stdCode.split(".")
            exchg = items[0]
            code = stdCode[(len(exchg)+1):]
            rq_code = stdCodeToRQ(stdCode)
            count += 1
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, stdCode, count, length))
            df_bars = rq.get_price(order_book_ids = rq_code,start_date=start_date, end_date=end_date,frequency=freq,adjust_type='none',expect_df=True)
            bars = list()
            total_nums = len(df_bars)
            cur_num = 0
            for idx, row in df_bars.iterrows():
                trade_date = row.name[1].to_pydatetime()
                date = int(trade_date.strftime("%Y%m%d"))
                if isDay:
                    time = 0
                else:
                    time = int(trade_date.strftime("%H%M"))
                curBar = {
                    "exchange":exchg,
                    "code": code,
                    "date": date,
                    "time": time,
                    "open": row["open"],
                    "high": row["open"],
                    "low": row["open"],
                    "close": row["open"],
                    "volume": row["volume"],
                    "turnover": row["total_turnover"]
                }

                if "settlement" in row:
                    curBar["settle"] = row["settlement"]

                if "open_interest" in row:
                    curBar["interest"] = row["open_interest"]

                bars.append(curBar)
                cur_num += 1
                if cur_num % 500 == 0:
                    print("Processing bars %d/%d..." % (cur_num, total_nums))

            print("Writing bars into database...")
            dbHelper.writeBars(bars, period)