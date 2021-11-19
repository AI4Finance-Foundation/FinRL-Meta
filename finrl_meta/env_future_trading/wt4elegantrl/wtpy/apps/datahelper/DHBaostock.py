from wtpy.apps.datahelper.DHDefs import BaseDataHelper, DBHelper
import baostock as bs
from datetime import datetime
import json
import os

def transCodes(codes:list) -> list:
    ret = list()
    for code in codes:
        items = code.split(".")
        exchg = items[0]
        if exchg == "SSE":
            ret.append("sh."+items[1])
        else:
            ret.append("sz."+items[1])

    return ret

class DHBaostock(BaseDataHelper):

    def __init__(self):
        BaseDataHelper.__init__(self)
        print("Baostock helper has been created.")
        return

    def auth(self, **kwargs):
        if self.isAuthed:
            return

        bs.login()
        self.isAuthed = True
        print("Baostock has been authorized.")

    def dmpCodeListToFile(self, filename:str, hasIndex:bool=True, hasStock:bool=True):
        raise Exception("Baostock has not code list api")

    def dmpAdjFactorsToFile(self, codes:list, filename:str):
        codes = transCodes(codes)
        stocks = {
            "SSE":{},
            "SZSE":{}
        }
        count = 0
        length = len(codes)
        for code in codes:
            exchg = code[:2]
            if exchg == 'sh':
                exchg = 'SSE'
            else:
                exchg = 'SZSE'
            count += 1

            stocks[exchg][code[3:]] = list()
            print("Fetching adjust factors of %s(%d/%s)..." % (code, count, length))
            rs = bs.query_adjust_factor(code=code, start_date="1990-01-01")

            if rs.error_code != '0':
                print("Error occured: %s" % (rs.error_msg))
                continue
    
            while rs.next():
                items = rs.get_row_data()
                date = int(items[1].replace("-",""))
                factor = float(items[4])
                stocks[exchg][code[3:]].append({
                    "date": date,
                    "factor": factor
                })
        
        print("Writing adjust factors into file %s..." % (filename))
        f = open(filename, 'w+')
        f.write(json.dumps(stocks, sort_keys=True, indent=4, ensure_ascii=False))
        f.close()

    def dmpBarsToFile(self, folder:str, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        codes = transCodes(codes)

        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        freq = ''
        isDay = False
        filetag = ''
        fields = ""
        if period == 'day':
            freq = 'd'
            isDay = True
            filetag = 'd'
            fields = "date,open,high,low,close,volume,amount"
        elif period == "min5":
            freq = '5'
            filetag = 'm5'
            fields = "date,time,open,high,low,close,volume,amount"
        else:
            raise Exception("Baostock has only bars of frequency day and min5")

        count = 0
        length = len(codes)
        for code in codes:
            exchg = code[:2]
            if exchg == 'sh':
                exchg = 'SSE'
            else:
                exchg = 'SZSE'
            count += 1
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, code, count, length))
            rs = bs.query_history_k_data_plus(code=code, fields=fields, start_date=start_date, end_date=end_date, frequency=freq)
            content = "date,time,open,high,low,close,volume,turnover\n"
            if rs.error_code != '0':
                print("Error occured: %s" % (rs.error_msg))
                continue

            while rs.next():
                items = rs.get_row_data().copy()
                if isDay:
                    items.insert(1, "0")
                else:
                    time = items[1][-9:-3]
                    items[1] = time[:2]+":"+time[2:4]+":"+time[4:]
                content += ",".join(items) + "\n"

            filename = "%s.%s_%s.csv" % (exchg, code[3:], filetag)
            filepath = os.path.join(folder, filename)
            print("Writing bars into file %s..." % (filepath))
            f = open(filepath, "w", encoding="utf-8")
            f.write(content)
            f.close()

    def dmpAdjFactorsToDB(self, dbHelper:DBHelper, codes:list):
        codes = transCodes(codes)
        stocks = {
            "SSE":{},
            "SZSE":{}
        }

        count = 0
        length = len(codes)
        for code in codes:
            exchg = code[:2]
            if exchg == 'sh':
                exchg = 'SSE'
            else:
                exchg = 'SZSE'
            count += 1
            
            print("Fetching adjust factors of %s(%d/%s)..." % (code, count, length))
            stocks[exchg][code[3:]] = list()
            rs = bs.query_adjust_factor(code=code, start_date="1990-01-01")

            if rs.error_code == '0':
                print("Error occured: %s" % (rs.error_msg))
                continue
    
            while rs.next():
                items = rs.get_row_data()
                date = int(items[1].replace("-",""))
                factor = float(items[4])
                stocks[exchg][code[3:]].append({
                    "date": date,
                    "factor": factor
                })
        
        print("Writing adjust factors into database...")
        dbHelper.writeFactors(stocks)

    def dmpBarsToDB(self, dbHelper:DBHelper, codes:list, start_date:datetime=None, end_date:datetime=None, period:str="day"):
        codes = transCodes(codes)

        if start_date is None:
            start_date = datetime(year=1990, month=1, day=1)
        
        if end_date is None:
            end_date = datetime.now()

        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        freq = ''
        isDay = False
        fields = ""
        if period == 'day':
            freq = 'd'
            isDay = True
            fields = "date,open,high,low,close,volume,amount"
        elif period == "min5":
            freq = '5'
            fields = "date,time,open,high,low,close,volume,amount"
        else:
            raise Exception("Baostock has only bars of frequency day and min5")

        count = 0
        length = len(codes)
        for code in codes:
            exchg = code[:2]
            if exchg == 'sh':
                exchg = 'SSE'
            else:
                exchg = 'SZSE'
            count += 1
            
            print("Fetching %s bars of %s(%d/%s)..." % (period, code, count, length))
            rs = bs.query_history_k_data_plus(code=code, fields=fields, start_date=start_date, end_date=end_date, frequency=freq)
            bars = []
            while (rs.error_code == '0') & rs.next():
                items = rs.get_row_data()
                if isDay:
                    bars.append({
                        "exchange":exchg,
                        "code":code[3:],
                        "date": int(items[0].replace("-","")),
                        "time": 0,
                        "open": float(items[1]),
                        "high": float(items[2]),
                        "low": float(items[3]),
                        "close": float(items[4]),
                        "volume": float(items[5]),
                        "turnover": float(items[6])
                    })
                else:
                    time = int(items[1][-9:-5])
                    bars.append({
                        "exchange":exchg,
                        "code":code[3:],
                        "date": int(items[0].replace("-","")),
                        "time": time,
                        "open": float(items[2]),
                        "high": float(items[3]),
                        "low": float(items[4]),
                        "close": float(items[5]),
                        "volume": float(items[6]),
                        "turnover": float(items[7])
                    })

            print("Writing bars into database...")
            dbHelper.writeBars(bars, period)