from wtpy.apps.datahelper.DHDefs import DBHelper
import pymysql
import math
import os

class MysqlHelper(DBHelper):
    def __init__(self, host:str, user:str, pwd:str, dbname:str, port:int=3306):
        self.params = {
            "host":host,
            'user':user,
            'password':pwd,
            'database':dbname,
            'port':port
        }
        self.conn:pymysql.Connection = None
    
    def __get_conn__(self):
        if self.conn is None:
            self.conn = pymysql.connect(**self.params)
        
        try:
            self.conn.ping()
        except:
            self.conn = pymysql.connect(**self.params)

        return self.conn 

    def initDB(self):
        paths = os.path.split(__file__)
        a = (paths[:-1] + ('initdb_mysql.sql',))
        _path = os.path.join(*a)
        f = open(_path, "r", encoding="UTF-8")
        content = f.read()
        f.close()
        conn = self.__get_conn__()
        cursor = conn.cursor()
        items = content.split(";")
        for item in items:
            item = item.strip()
            if len(item) == 0:
                continue
            cursor.execute(item+";")
        conn.commit()
        cursor.close()

    def writeBars(self, bars:list, period="day"):
        count = 0
        sql = ""
        isDay = (period=='day')
        tbname = "tb_kline_%s" % (period)
        for curBar in bars:
            if count == 0:
                if isDay:
                    sql = "REPLACE INTO %s(exchange,`code`,`date`,open,high,low,close,settle,volume,turnover,interest,diff_interest) VALUES" % (tbname)
                else:
                    sql = "REPLACE INTO %s(exchange,`code`,`date`,`time`,open,high,low,close,volume,turnover,interest,diff_interest) VALUES" % (tbname)
            
            if isDay:
                subsql = "('%s','%s',%d,%f,%f,%f,%f," % (curBar["exchange"], curBar["code"], curBar["date"], curBar["open"], curBar["high"], curBar["low"], curBar["close"])
                if "settle" in curBar:
                    subsql += str(curBar["settle"]) + ","
                else:
                    subsql += "0,"
                if "volume" in curBar:
                    subsql += str(curBar["volume"]) + ","
                else:
                    subsql += "0,"
                if "turnover" in curBar:
                    subsql += str(curBar["turnover"]) + ","
                else:
                    subsql += "0,"
                if "interest" in curBar:
                    subsql += str(curBar["interest"]) + ","
                else:
                    subsql += "0,"
                if "diff_interest" in curBar:
                    subsql += str(curBar["diff_interest"]) + ","
                else:
                    subsql += "0,"
                subsql = subsql[:-1] + "),"
                sql += subsql
            else:
                barTime = (curBar["date"] - 19900000)*10000 + curBar["time"]
                subsql = "('%s','%s',%d,%d,%f,%f,%f,%f," % (curBar["exchange"], curBar["code"], curBar["date"], barTime, curBar["open"], curBar["high"], curBar["low"], curBar["close"])
                if "volume" in curBar:
                    subsql += str(curBar["volume"]) + ","
                else:
                    subsql += "0,"
                if "turnover" in curBar:
                    subsql += str(curBar["turnover"]) + ","
                else:
                    subsql += "0,"
                if "interest" in curBar:
                    subsql += str(curBar["interest"]) + ","
                else:
                    subsql += "0,"
                if "diff_interest" in curBar:
                    subsql += str(curBar["diff_interest"]) + ","
                else:
                    subsql += "0,"
                subsql = subsql[:-1] + "),"
                sql += subsql

            count += 1
            if count == 500:
                count = 0
                sql = sql[:-1] + ";"
                conn = self.__get_conn__()
                cursor = conn.cursor()
                cursor.execute(sql)
                conn.commit()
                cursor.close()

        # 循环完了，再做一次提交
        if count > 0:
            sql = sql[:-1] + ";"
            conn = self.__get_conn__()
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            cursor.close()


    def writeFactors(self, factors:dict):
        for exchg in factors:
            codelist = factors[exchg]
            for code in codelist:
                items = codelist[code]
                sql = 'REPLACE INTO tb_adj_factors(exchange,`code`,`date`,factor) VALUES'
                for item in items:
                    sql += "('%s','%s',%d,%f)," % (exchg, code, item["date"], item["factor"])

                sql = sql[:-1] + ";"
                conn = self.__get_conn__()
                cursor = conn.cursor()
                cursor.execute(sql)
                conn.commit()
                cursor.close()