import json
import os
import sqlite3
import hashlib
import datetime
from .WtLogger import WtLogger

def backup_file(filename):
    if not os.path.exists(filename):
        return

    items = filename.split(".")
    ext = items[-1]
    prefix = ".".join(items[:-1])

    now = datetime.datetime.now()
    timetag = now.strftime("%Y%m%d_%H%M%S")
    target = prefix + "_" + timetag + "." + ext
    import shutil
    shutil.copy(filename, target)

class DataMgr:

    def __init__(self, datafile:str="mondata.db", logger:WtLogger=None):
        self.__grp_cache__ = dict()
        self.__logger__ = logger

        self.__db_conn__ = sqlite3.connect(datafile, check_same_thread=False)
        self.__check_db__()

        #加载组合列表
        cur = self.__db_conn__.cursor()
        self.__config__ = {
            "groups":{},
            "users":{}
        }

        for row in cur.execute("SELECT * FROM groups;"):
            grpInfo = dict()
            grpInfo["id"] = row[1]
            grpInfo["name"] = row[2]
            grpInfo["path"] = row[3]
            grpInfo["info"] = row[4]
            grpInfo["gtype"] = row[5]
            grpInfo["datmod"] = row[6]
            grpInfo["env"] = row[7]
            grpInfo["mqurl"] = row[8]
            self.__config__["groups"][grpInfo["id"]] = grpInfo

        for row in cur.execute("SELECT * FROM users;"):
            usrInfo = dict()
            usrInfo["loginid"] = row[1]
            usrInfo["name"] = row[2]
            usrInfo["role"] = row[3]
            usrInfo["passwd"] = row[4]
            usrInfo["iplist"] = row[5]
            usrInfo["remark"] = row[6]
            usrInfo["createby"] = row[7]
            usrInfo["createtime"] = row[8]
            usrInfo["modifyby"] = row[9]
            usrInfo["modifytime"] = row[10]
            self.__config__["users"][usrInfo["loginid"]] = usrInfo

    def get_db(self):
        return self.__db_conn__

    def __check_db__(self):
        if self.__db_conn__ is None:
            return

        cur = self.__db_conn__.cursor()
        tables = []
        for row in cur.execute("select name from sqlite_master where type='table' order by name"):
            tables.append(row[0])
        
        if "actions" not in tables:
            sql = "CREATE TABLE [actions] (\n"
            sql += "[id] INTEGER PRIMARY KEY autoincrement, \n"
            sql += "[loginid] VARCHAR(20) NOT NULL DEFAULT '', \n"
            sql += "[actiontime] DATETIME default (datetime('now', 'localtime')), \n"
            sql += "[actionip] VARCHAR(30) NOT NULL DEFAULT '', \n"
            sql += "[actiontype] VARCHAR(20) NOT NULL DEFAULT '',\n"
            sql += "[remark] TEXT default '');"
            cur.execute(sql)
            cur.execute("CREATE INDEX [idx_actions_loginid] ON [actions] ([loginid]);")
            cur.execute("CREATE INDEX [idx_actions_actiontime] ON [actions] ([actiontime]);")
            self.__db_conn__.commit()

        if "groups" not in tables:
            sql = "CREATE TABLE [groups] (\n"
            sql += "[id] INTEGER PRIMARY KEY autoincrement,\n"
            sql += "[groupid] VARCHAR(20) NOT NULL DEFAULT '',\n"
            sql += "[name] VARCHAR(30) NOT NULL DEFAULT '',\n"
            sql += "[path] VARCHAR(256) NOT NULL DEFAULT '',\n"
            sql += "[info] TEXT DEFAULT '',\n"
            sql += "[gtype] VARCHAR(10) NOT NULL DEFAULT 'cta',\n"
            sql += "[datmod] VARCHAR(10) NOT NULL DEFAULT 'mannual',\n"
            sql += "[env] VARCHAR(20) NOT NULL DEFAULT 'product',\n"
            sql += "[mqurl] VARCHAR(255) NOT NULL DEFAULT '',\n"
            sql += "[createtime] DATETIME default (datetime('now', 'localtime')),\n"
            sql += "[modifytime] DATETIME default (datetime('now', 'localtime')));"
            cur.execute(sql)
            cur.execute("CREATE UNIQUE INDEX [idx_groupid] ON [groups] ([groupid]);")
            self.__db_conn__.commit()

        if "schedules" not in tables:
            sql = "CREATE TABLE [schedules] (\n"
            sql += "[id] INTEGER PRIMARY KEY autoincrement,\n"
            sql += "[appid] VARCHAR(20) NOT NULL DEFAULT '',\n"
            sql += "[path] VARCHAR(256) NOT NULL DEFAULT '',\n"
            sql += "[folder] VARCHAR(256) NOT NULL DEFAULT '',\n"
            sql += "[param] VARCHAR(50) NOT NULL DEFAULT '',\n"
            sql += "[type] INTEGER DEFAULT 0,\n"
            sql += "[span] INTEGER DEFAULT 3,\n"
            sql += "[guard] VARCHAR(20) DEFAULT 'false',\n"
            sql += "[redirect] VARCHAR(20) DEFAULT 'false',\n"
            sql += "[schedule] VARCHAR(20) DEFAULT 'false',\n"
            sql += "[weekflag] VARCHAR(20) DEFAULT '000000',\n"
            sql += "[mqurl] VARCHAR(255) NOT NULL DEFAULT '',\n"
            sql += "[task1] VARCHAR(100) NOT NULL DEFAULT '{\"active\": true,\"time\": 0,\"action\": 0}',\n"
            sql += "[task2] VARCHAR(100) NOT NULL DEFAULT '{\"active\": true,\"time\": 0,\"action\": 0}',\n"
            sql += "[task3] VARCHAR(100) NOT NULL DEFAULT '{\"active\": true,\"time\": 0,\"action\": 0}',\n"
            sql += "[task4] VARCHAR(100) NOT NULL DEFAULT '{\"active\": true,\"time\": 0,\"action\": 0}',\n"
            sql += "[task5] VARCHAR(100) NOT NULL DEFAULT '{\"active\": true,\"time\": 0,\"action\": 0}',\n"
            sql += "[task6] VARCHAR(100) NOT NULL DEFAULT '{\"active\": true,\"time\": 0,\"action\": 0}',\n"
            sql += "[createtime] DATETIME default (datetime('now', 'localtime')),\n"
            sql += "[modifytime] DATETIME default (datetime('now', 'localtime')));"
            cur.execute(sql)
            cur.execute("CREATE UNIQUE INDEX [idx_appid] ON [schedules] ([appid]);")
            self.__db_conn__.commit()

        if "users" not in tables:
            sql = "CREATE TABLE [users] (\n"
            sql += "[id] INTEGER PRIMARY KEY autoincrement,\n"
            sql += "[loginid] VARCHAR(20) NOT NULL DEFAULT '',\n"
            sql += "[name] VARCHAR(30) NOT NULL DEFAULT '',\n"
            sql += "[role] VARCHAR(10) NOT NULL DEFAULT '',\n"
            sql += "[passwd] VARCHAR(30) NOT NULL DEFAULT 'cta',\n"
            sql += "[iplist] VARCHAR(100) NOT NULL DEFAULT 'mannual',\n"
            sql += "[remark] VARCHAR(256) NOT NULL DEFAULT '',\n"
            sql += "[createby] VARCHAR(20) NOT NULL DEFAULT '',\n"
            sql += "[createtime] DATETIME default (datetime('now', 'localtime')),\n"
            sql += "[modifyby] VARCHAR(20) NOT NULL DEFAULT '',\n"
            sql += "[modifytime] DATETIME default (datetime('now', 'localtime')));"
            cur.execute(sql)
            cur.execute("CREATE UNIQUE INDEX [idx_loginid] ON [users] ([loginid]);")
            self.__db_conn__.commit()

    def __check_cache__(self, grpid, grpInfo):
        now = datetime.datetime.now()
        if grpid not in self.__grp_cache__:
            self.__grp_cache__[grpid] = dict()
            self.__grp_cache__[grpid]["cachetime"] = None
        else:
            cache_time = self.__grp_cache__[grpid]["cachetime"]
            bNeedReset = False
            if cache_time is None:
                bNeedReset = True
            else:
                td = now - cache_time
                if td.total_seconds() >= 60:# 上次缓存时间超过60s，则重新读取
                    bNeedReset = True

            if bNeedReset:
                self.__grp_cache__[grpid] = dict()
                self.__grp_cache__[grpid]["cachetime"] = None

        if "strategies" not in self.__grp_cache__[grpid]:
            filepath = "./generated/marker.json"
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                try:
                    f = open(filepath, "r")
                    content = f.read()
                    marker = json.loads(content)
                    f.close()

                    self.__grp_cache__[grpid] = {
                        "strategies":marker["marks"],
                        "channels":marker["channels"]
                    } 

                    if "executers" in marker:
                        self.__grp_cache__[grpid]["executers"] = marker["executers"]
                    else:
                        self.__grp_cache__[grpid]["executers"] = []

                except:
                    self.__grp_cache__[grpid] = {
                        "strategies":[],
                        "channels":[],
                        "executers":[]
                    } 
            self.__grp_cache__[grpid]["strategies"].sort()
            self.__grp_cache__[grpid]["channels"].sort()
            self.__grp_cache__[grpid]["executers"].sort()
            self.__grp_cache__[grpid]["cachetime"] = now

    def get_groups(self, tpfilter:str=''):
        ret = []
        for grpid in self.__config__["groups"]:
            grpinfo = self.__config__["groups"][grpid]
            if tpfilter == '':
                ret.append(grpinfo)
            elif grpinfo["gtype"] == tpfilter:
                ret.append(grpinfo)
        
        return ret

    def has_group(self, grpid:str):
        return (grpid in self.__config__["groups"])

    def get_group(self, grpid:str):
        if grpid in self.__config__["groups"]:
            return self.__config__["groups"][grpid]
        else:
            return None

    def get_group_cfg(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return "{}"
        else:
            grpInfo = self.__config__["groups"][grpid]
            filepath = "./config.json"
            filepath = os.path.join(grpInfo["path"], filepath)
            f = open(filepath, "r")
            content = f.read()
            f.close()
            return json.loads(content)

    def set_group_cfg(self, grpid:str, config:dict):
        if grpid not in self.__config__["groups"]:
            return False
        else:
            grpInfo = self.__config__["groups"][grpid]
            filepath = "./config.json"
            filepath = os.path.join(grpInfo["path"], filepath)
            backup_file(filepath)
            f = open(filepath, "w")
            f.write(json.dumps(config, indent=4))
            f.close()
            return True

    def get_group_entry(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return "{}"
        else:
            grpInfo = self.__config__["groups"][grpid]
            filepath = "./run.py"
            filepath = os.path.join(grpInfo["path"], filepath)
            f = open(filepath, "r", encoding="utf-8")
            content = f.read()
            f.close()
            return content

    def set_group_entry(self, grpid:str, content:str):
        if grpid not in self.__config__["groups"]:
            return False
        else:
            grpInfo = self.__config__["groups"][grpid]
            filepath = "./run.py"
            filepath = os.path.join(grpInfo["path"], filepath)
            backup_file(filepath)
            f = open(filepath, "w", encoding="utf-8")
            f.write(content)
            f.close()
            return True

    def add_group(self, grpInfo:dict):
        grpid = grpInfo["id"]
        isNewGrp = not (grpid in self.__config__["groups"])

        bSucc = False
        try:
            cur = self.__db_conn__.cursor()
            sql = ''
            if isNewGrp:
                sql = "INSERT INTO groups(groupid,name,path,info,gtype,datmod,env,mqurl) VALUES('%s','%s','%s','%s','%s','%s','%s','%s');" \
                    % (grpid, grpInfo["name"], grpInfo["path"], grpInfo["info"], grpInfo["gtype"], grpInfo["datmod"], grpInfo["env"], grpInfo["mqurl"])
            else:
                sql = "UPDATE groups SET name='%s',path='%s',info='%s',gtype='%s',datmod='%s',env='%s',mqurl='%s',modifytime=datetime('now','localtime') WHERE groupid='%s';" \
                    % (grpInfo["name"], grpInfo["path"], grpInfo["info"], grpInfo["gtype"], grpInfo["datmod"], grpInfo["env"], grpInfo["mqurl"], grpid)
            cur.execute(sql)
            self.__db_conn__.commit()
            bSucc = True
        except sqlite3.Error as e:
            print(e)

        if bSucc:
            self.__config__["groups"][grpid] = grpInfo

        return bSucc

    def del_group(self, grpid:str):
        if grpid in self.__config__["groups"]:
            self.__config__["groups"].pop(grpid)
            
            cur = self.__db_conn__.cursor()
            cur.execute("DELETE FROM groups WHERE groupid='%s';" % (grpid))
            self.__db_conn__.commit()

    def get_users(self):
        ret = []
        for loginid in self.__config__["users"]:
            usrInfo = self.__config__["users"][loginid]
            ret.append(usrInfo.copy())                
        
        return ret

    def add_user(self, usrInfo, admin):
        loginid = usrInfo["loginid"]
        isNewUser = not (loginid in self.__config__["users"])

        cur = self.__db_conn__.cursor()
        now = datetime.datetime.now()
        if isNewUser:
            encpwd = hashlib.md5((loginid+usrInfo["passwd"]).encode("utf-8")).hexdigest()
            usrInfo["passwd"] = encpwd
            usrInfo["createby"] = admin
            usrInfo["modifyby"] = admin
            usrInfo["createtime"] = now.strftime("%Y-%m-%d %H:%M:%S")
            usrInfo["modifytime"] = now.strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("INSERT INTO users(loginid,name,role,passwd,iplist,remark,createby,modifyby) VALUES(?,?,?,?,?,?,?,?);", 
                (loginid, usrInfo["name"], usrInfo["role"], encpwd, usrInfo["iplist"], usrInfo["remark"], admin, admin))
        else:
            usrInfo["modifyby"] = admin
            usrInfo["modifytime"] = now.strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("UPDATE users SET name=?,role=?,iplist=?,remark=?,modifyby=?,modifytime=datetime('now','localtime') WHERE loginid=?;", 
                (usrInfo["name"], usrInfo["role"], usrInfo["iplist"], usrInfo["remark"], admin, loginid))
        self.__db_conn__.commit()

        self.__config__["users"][loginid] = usrInfo

    def mod_user_pwd(self, loginid:str, newpwd:str, admin:str):
        cur = self.__db_conn__.cursor()
        cur.execute("UPDATE users SET passwd=?,modifyby=?,modifytime=datetime('now','localtime') WHERE loginid=?;", 
                (newpwd,admin,loginid))
        self.__db_conn__.commit()
        self.__config__["users"][loginid]["passwd"]=newpwd


    def del_user(self, loginid, admin):
        if loginid in self.__config__["users"]:
            self.__config__["users"].pop(loginid)
            
            cur = self.__db_conn__.cursor()
            cur.execute("DELETE FROM users WHERE loginid='%s';" % (loginid))
            self.__db_conn__.commit()
            return True
        else:
            return False

    def log_action(self, adminInfo, atype, remark):
        cur = self.__db_conn__.cursor()
        sql = "INSERT INTO actions(loginid,actiontime,actionip,actiontype,remark) VALUES('%s',datetime('now','localtime'),'%s','%s','%s');" % (
                adminInfo["loginid"], adminInfo["loginip"], atype, remark)
        cur.execute(sql)
        self.__db_conn__.commit()

    def get_user(self, loginid:str):
        if loginid in self.__config__["users"]:
            return self.__config__["users"][loginid].copy()
        elif loginid == 'superman':
            return {
                "loginid":loginid,
                "name":"超管",
                "role":"superman",
                "passwd":"25ed305a56504e95fd1ca9900a1da174",
                "iplist":"",
                "remark":"内置超管账号",
                'builtin':True
            }
        else:
            return None

    def get_strategies(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
        
        return self.__grp_cache__[grpid]["strategies"]

    def get_channels(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
        
        return self.__grp_cache__[grpid]["channels"]

    def get_trades(self, grpid:str, straid:str, limit:int = 200):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        if straid not in self.__grp_cache__[grpid]["strategies"]:
            return []

        if "trades" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["trades"] = dict()
        
        if straid not in self.__grp_cache__[grpid]["trades"]:
            filepath = "./generated/outputs/%s/trades.csv" % (straid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                trdCache = dict()
                trdCache["file"] = filepath
                trdCache["lastrow"] = 0
                trdCache["trades"] = list()
                self.__grp_cache__[grpid]["trades"][straid] = trdCache

        trdCache = self.__grp_cache__[grpid]["trades"][straid]
        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")
            if len(cells) > 10:
                continue

            tItem = {
                "strategy":straid,
                "code": cells[0],
                "time": int(cells[1]),
                "direction": cells[2],
                "offset": cells[3],
                "price": float(cells[4]),
                "volume": float(cells[5]),
                "tag": cells[6],
                "fee": 0
            }

            if len(cells) > 7:
                tItem["fee"] = float(cells[7])

            trdCache["trades"].append(tItem)
            trdCache["lastrow"] += 1
        
        return trdCache["trades"][-limit:]

    def get_funds(self, grpid:str, straid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        if straid not in self.__grp_cache__[grpid]["strategies"]:
            return []

        if "funds" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["funds"] = dict()
        
        if straid not in self.__grp_cache__[grpid]["funds"]:
            filepath = "./generated/outputs/%s/funds.csv" % (straid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                trdCache = dict()
                trdCache["file"] = filepath
                trdCache["lastrow"] = 0
                trdCache["funds"] = list()
                self.__grp_cache__[grpid]["funds"][straid] = trdCache

        trdCache = self.__grp_cache__[grpid]["funds"][straid]

        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")
            if len(cells) > 10:
                continue

            tItem = {
                "strategy":straid,
                "date": int(cells[0]),
                "closeprofit": float(cells[1]),
                "dynprofit": float(cells[2]),
                "dynbalance": float(cells[3]),
                "fee": 0
            }

            if len(cells) > 4:
                tItem["fee"] = float(cells[4])

            trdCache["funds"].append(tItem)
            trdCache["lastrow"] += 1

        ret = trdCache["funds"].copy()

        if len(ret) > 0:
            last_date = ret[-1]["date"]
        else:
            last_date = 0

        # 这里再更新一条实时数据
        filepath = "./generated/stradata/%s.json" % (straid)
        filepath = os.path.join(grpInfo["path"], filepath)
        f = open(filepath, "r")
        try:
            content = f.read()
            json_data = json.loads(content)
            fund = json_data["fund"]
            if fund["tdate"] > last_date:
                ret.append({
                    "strategy":straid,
                    "date": fund["tdate"],
                    "closeprofit": fund["total_profit"],
                    "dynprofit": fund["total_dynprofit"],
                    "dynbalance": fund["total_profit"] + fund["total_dynprofit"] - fund["total_fees"],
                    "fee": fund["total_fees"]
                })
        except:
            pass
        f.close()
        
        return ret

    def get_signals(self, grpid:str, straid:str, limit:int = 200):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        if straid not in self.__grp_cache__[grpid]["strategies"]:
            return []

        if "signals" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["signals"] = dict()
        
        if straid not in self.__grp_cache__[grpid]["signals"]:
            filepath = "./generated/outputs/%s/signals.csv" % (straid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                trdCache = dict()
                trdCache["file"] = filepath
                trdCache["lastrow"] = 0
                trdCache["signals"] = list()
                self.__grp_cache__[grpid]["signals"][straid] = trdCache

        trdCache = self.__grp_cache__[grpid]["signals"][straid]

        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")

            tItem = {
                "strategy":straid,
                "code": cells[0],
                "target": float(cells[1]),
                "sigprice": float(cells[2]),
                "gentime": cells[3],
                "tag": cells[4]
            }

            trdCache["signals"].append(tItem)

        trdCache["lastrow"] += len(lines)        
        return trdCache["signals"][-limit:]

    def get_rounds(self, grpid:str, straid:str, limit:int = 200):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        if straid not in self.__grp_cache__[grpid]["strategies"]:
            return []

        if "rounds" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["rounds"] = dict()
        
        if straid not in self.__grp_cache__[grpid]["rounds"]:
            filepath = "./generated/outputs/%s/closes.csv" % (straid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                trdCache = dict()
                trdCache["file"] = filepath
                trdCache["lastrow"] = 0
                trdCache["rounds"] = list()
                self.__grp_cache__[grpid]["rounds"][straid] = trdCache

        trdCache = self.__grp_cache__[grpid]["rounds"][straid]
        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")

            tItem = {
                "strategy":straid,
                "code": cells[0],
                "direct": cells[1],
                "opentime": int(cells[2]),
                "openprice": float(cells[3]),
                "closetime": int(cells[4]),
                "closeprice": float(cells[5]),
                "qty": float(cells[6]),
                "profit": float(cells[7]),
                "entertag": cells[9],
                "exittag": cells[10]
            }

            trdCache["rounds"].append(tItem)
        trdCache["lastrow"] += len(lines)
        
        return trdCache["rounds"][-limit:]

    def get_positions(self, grpid:str, straid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        ret = list()
        if straid != "all":
            if straid not in self.__grp_cache__[grpid]["strategies"]:
                return []
            
            filepath = "./generated/stradata/%s.json" % (straid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            
            f = open(filepath, "r")
            try:
                content = f.read()
                json_data = json.loads(content)

                positions = json_data["positions"]
                for pItem in positions:
                    tag = "volumn" if "volume" not in pItem else "volume"
                    if pItem[tag] == 0.0:
                        continue

                    for dItem in pItem["details"]:
                        dItem["code"] = pItem["code"]
                        dItem["strategy"] = straid
                        if "volumn" in dItem:
                            dItem["volume"] = dItem["volumn"]
                            dItem.pop("volumn")
                        ret.append(dItem)
            except:
                pass

            f.close()
        else:
            for straid in self.__grp_cache__[grpid]["strategies"]:
                filepath = "./generated/stradata/%s.json" % (straid)
                filepath = os.path.join(grpInfo["path"], filepath)
                if not os.path.exists(filepath):
                    return []
                
                f = open(filepath, "r")
                try:
                    content = f.read()
                    json_data = json.loads(content)

                    positions = json_data["positions"]
                    for pItem in positions:
                        tag = "volumn" if "volume" not in pItem else "volume"
                        if pItem[tag] == 0.0:
                            continue

                        for dItem in pItem["details"]:
                            dItem["code"] = pItem["code"]
                            dItem["strategy"] = straid
                            if "volumn" in dItem:
                                dItem["volume"] = dItem["volumn"]
                                dItem.pop("volumn")
                            ret.append(dItem)
                except:
                    pass

                f.close()
        return ret

    def get_channel_orders(self, grpid:str, chnlid:str, limit:int = 200):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        if chnlid not in self.__grp_cache__[grpid]["channels"]:
            return []

        if "corders" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["corders"] = dict()
        
        if chnlid not in self.__grp_cache__[grpid]["corders"]:
            filepath = "./generated/traders/%s/orders.csv" % (chnlid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                trdCache = dict()
                trdCache["file"] = filepath
                trdCache["lastrow"] = 0
                trdCache["corders"] = list()
                self.__grp_cache__[grpid]["corders"][chnlid] = trdCache

        trdCache = self.__grp_cache__[grpid]["corders"][chnlid]

        f = open(trdCache["file"], "r",encoding="gb2312",errors="ignore")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")

            tItem = {
                "channel":chnlid,
                "localid":int(cells[0]),
                "time":int(cells[2]),
                "code": cells[3],
                "action": cells[4],
                "total": float(cells[5]),
                "traded": float(cells[6]),
                "price": float(cells[7]),
                "orderid": cells[8],
                "canceled": cells[9],
                "remark": cells[10]
            }

            trdCache["corders"].append(tItem)
        
        return trdCache["corders"][-limit:]

    def get_channel_trades(self, grpid:str, chnlid:str, limit:int = 200):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        if chnlid not in self.__grp_cache__[grpid]["channels"]:
            return []

        if "ctrades" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["ctrades"] = dict()
        
        if chnlid not in self.__grp_cache__[grpid]["ctrades"]:
            filepath = "./generated/traders/%s/trades.csv" % (chnlid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            else:
                trdCache = dict()
                trdCache["file"] = filepath
                trdCache["lastrow"] = 0
                trdCache["ctrades"] = list()
                self.__grp_cache__[grpid]["ctrades"][chnlid] = trdCache

        trdCache = self.__grp_cache__[grpid]["ctrades"][chnlid]

        f = open(trdCache["file"], "r",encoding="gb2312")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")

            tItem = {
                "channel":chnlid,
                "localid":int(cells[0]),
                "time":int(cells[2]),
                "code": cells[3],
                "action": cells[4],
                "volume": float(cells[5]),
                "price": float(cells[6]),
                "tradeid": cells[7],
                "orderid": cells[8]
            }

            trdCache["ctrades"].append(tItem)
        
        return trdCache["ctrades"][-limit:]

    def get_channel_positions(self, grpid:str, chnlid:str):
        if self.__config__ is None:
            return []

        if "groups" not in self.__config__:
            return []

        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        ret = list()
        channels = list()
        if chnlid != 'all':
            channels.append(chnlid)
        else:
            channels = self.__grp_cache__[grpid]["channels"]

        for cid in channels:
            if cid not in self.__grp_cache__[grpid]["channels"]:
                continue
            
            filepath = "./generated/traders/%s/rtdata.json" % (cid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                return []
            
            f = open(filepath, "r")
            try:
                content = f.read()
                json_data = json.loads(content)

                positions = json_data["positions"]
                for pItem in positions:
                    pItem["channel"] = cid
                    ret.append(pItem)
            except:
                pass

            f.close()
        return ret

    def get_channel_funds(self, grpid:str, chnlid:str):
        if self.__config__ is None:
            return []

        if "groups" not in self.__config__:
            return []

        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
            
        ret = dict()
        channels = list()
        if chnlid != 'all':
            channels.append(chnlid)
        else:
            channels = self.__grp_cache__[grpid]["channels"]
            print(channels)

        for cid in channels:
            if cid not in self.__grp_cache__[grpid]["channels"]:
                continue
            
            filepath = "./generated/traders/%s/rtdata.json" % (cid)
            filepath = os.path.join(grpInfo["path"], filepath)
            if not os.path.exists(filepath):
                continue
            
            f = open(filepath, "r")
            try:
                content = f.read()
                json_data = json.loads(content)

                funds = json_data["funds"]
                ret[cid] = funds
            except:
                pass

            f.close()
        print(ret)
        return ret

    def get_actions(self, sdate, edate):
        ret = list()

        cur = self.__db_conn__.cursor()
        for row in cur.execute("SELECT id,loginid,actiontime,actionip,actiontype,remark FROM actions WHERE actiontime>=? and actiontime<=?;", (sdate, edate)):
            aInfo = dict()
            aInfo["id"] = row[0]
            aInfo["loginid"] = row[1]
            aInfo["actiontime"] = row[2]
            aInfo["actionip"] = row[3]
            aInfo["action"] = row[4]
            aInfo["remark"] = row[5]

            ret.append(aInfo)

        return ret

    def get_group_trades(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)

        if "grptrades" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["grptrades"] = dict()
        
        filepath = "./generated/portfolio/trades.csv"
        filepath = os.path.join(grpInfo["path"], filepath)
        print(filepath)
        if not os.path.exists(filepath):
            return []
        else:
            trdCache = dict()
            trdCache["file"] = filepath
            trdCache["lastrow"] = 0
            trdCache["trades"] = list()
            self.__grp_cache__[grpid]["grptrades"]["cache"] = trdCache

        trdCache = self.__grp_cache__[grpid]["grptrades"]['cache']

        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]
        print(lines)

        for line in lines:
            cells = line.split(",")

            tItem = {
                "code": cells[0],
                "time": int(cells[1]),
                "direction": cells[2],
                "offset": cells[3],
                "price": float(cells[4]),
                "volume": float(cells[5]),
                "fee": float(cells[6])
            }

            trdCache["trades"].append(tItem)
            trdCache["lastrow"] += 1
        
        return trdCache["trades"]

    def get_group_rounds(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)

        if "grprounds" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["grprounds"] = dict()
        
        filepath = "./generated/portfolio/closes.csv"
        filepath = os.path.join(grpInfo["path"], filepath)
        if not os.path.exists(filepath):
            return []
        else:
            trdCache = dict()
            trdCache["file"] = filepath
            trdCache["lastrow"] = 0
            trdCache["rounds"] = list()
            self.__grp_cache__[grpid]["grprounds"]["cache"] = trdCache

        trdCache = self.__grp_cache__[grpid]["grprounds"]['cache']

        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]
        print(lines)

        for line in lines:
            cells = line.split(",")

            tItem = {
                "code": cells[0],
                "direct": cells[1],
                "opentime": int(cells[2]),
                "openprice": float(cells[3]),
                "closetime": int(cells[4]),
                "closeprice": float(cells[5]),
                "qty": float(cells[6]),
                "profit": float(cells[7])
            }

            trdCache["rounds"].append(tItem)
            trdCache["lastrow"] += 1
        
        return trdCache["rounds"]

    def get_group_funds(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)

        if "grpfunds" not in self.__grp_cache__[grpid]:
            self.__grp_cache__[grpid]["grpfunds"] = dict()
        
        filepath = "./generated/portfolio/funds.csv"
        filepath = os.path.join(grpInfo["path"], filepath)
        if not os.path.exists(filepath):
            return []
        else:
            trdCache = dict()
            trdCache["file"] = filepath
            trdCache["lastrow"] = 0
            trdCache["funds"] = list()
            self.__grp_cache__[grpid]["grpfunds"]["cache"] = trdCache

        trdCache = self.__grp_cache__[grpid]["grpfunds"]['cache']

        f = open(trdCache["file"], "r")
        last_row = trdCache["lastrow"]
        lines = f.readlines()
        f.close()
        lines = lines[1+last_row:]

        for line in lines:
            cells = line.split(",")

            tItem = {
                "date": int(cells[0]),
                "predynbalance": float(cells[1]),
                "prebalance": float(cells[2]),
                "balance": float(cells[3]),
                "closeprofit": float(cells[4]),
                "dynprofit": float(cells[5]),
                "fee": float(cells[6]),
                "maxdynbalance": float(cells[7]),
                "maxtime": float(cells[8]),
                "mindynbalance": float(cells[9]),
                "mintime": float(cells[10]),
                "mdmaxbalance": float(cells[11]),
                "mdmaxdate": float(cells[12]),
                "mdminbalance": float(cells[13]),
                "mdmindate": float(cells[14])
            }

            trdCache["funds"].append(tItem)
            trdCache["lastrow"] += 1
        
        ret = trdCache["funds"].copy()

        if len(ret) > 0:
            last_date = ret[-1]["date"]
        else:
            last_date = 0

        # 这里再更新一条实时数据
        filepath = "./generated/portfolio/datas.json"
        filepath = os.path.join(grpInfo["path"], filepath)
        f = open(filepath, "r")
        try:
            content = f.read()
            json_data = json.loads(content)
            fund = json_data["fund"]
            if ["date"] > last_date:
                ret.append({
                    "date": fund["date"],
                    "predynbalance": fund["predynbal"],
                    "prebalance": fund["prebalance"],
                    "balance": fund["balance"],
                    "closeprofit": fund["profit"],
                    "dynprofit": fund["dynprofit"],
                    "fee": fund["fees"],
                    "maxdynbalance": fund["max_dyn_bal"],
                    "maxtime": fund["max_time"],
                    "mindynbalance": fund["min_dyn_bal"],
                    "mintime": fund["min_time"],
                    "mdmaxbalance": fund["maxmd"]["dyn_balance"],
                    "mdmaxdate": fund["maxmd"]["date"],
                    "mdminbalance": fund["minmd"]["dyn_balance"],
                    "mdmindate": fund["minmd"]["date"]
                })
        except:
            pass
        f.close()
        return ret

    def get_group_positions(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return []

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
        
        filepath = "./generated/portfolio/datas.json"
        filepath = os.path.join(grpInfo["path"], filepath)
        if not os.path.exists(filepath):
            return []
        else:
            ret = list()
            f = open(filepath, "r")
            try:
                content = f.read()
                json_data = json.loads(content)

                positions = json_data["positions"]
                for pItem in positions:
                    if pItem["volume"] == 0:
                        continue

                    for dItem in pItem["details"]:
                        dItem["code"] = pItem["code"]
                        ret.append(dItem)
            except:
                pass

            f.close()
            return ret

    def get_group_performances(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return {}

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
        
        filepath = "./generated/portfolio/datas.json" 
        filepath = os.path.join(grpInfo["path"], filepath)
        if not os.path.exists(filepath):
            return {}
        else:
            perf = dict()
            f = open(filepath, "r")
            try:
                content = f.read()
                json_data = json.loads(content)

                positions = json_data["positions"]
                for pItem in positions:
                    code = pItem['code']
                    ay = code.split(".")
                    pid = code
                    if len(ay) > 2:
                        if ay[1] not in ['IDX','STK','ETF']:
                            pid = ay[0] + "." + ay[1]
                        else:
                            pid = ay[0] + "." + ay[2]

                    if pid not in perf:
                        perf[pid] = {
                            'closeprofit':0,
                            'dynprofit':0
                        }

                    perf[pid]['closeprofit'] += pItem['closeprofit']
                    perf[pid]['dynprofit'] += pItem['dynprofit']
                    
            except:
                pass

            f.close()
            return perf

    def get_group_filters(self, grpid:str):
        if grpid not in self.__config__["groups"]:
            return {}

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)
        
        filepath = os.path.join(grpInfo["path"], 'filters.json')
        if not os.path.exists(filepath):
            filters = {}
        else:
            filters = {}
            f = open(filepath, "r")
            try:
                content = f.read()
                filters = json.loads(content)
            except:
                pass

            f.close()

        gpCache = self.__grp_cache__[grpid]
        if "executer_filters" not in filters:
            filters["executer_filters"] = dict()
        if "strategy_filters" not in filters:
            filters["strategy_filters"] = dict()
        if "code_filters" not in filters:
            filters["code_filters"] = dict()

        for sid in gpCache["strategies"]:
            if sid not in filters['strategy_filters']:
                filters['strategy_filters'][sid] = False
        
        for eid in gpCache["executers"]:
            if eid not in filters['executer_filters']:
                filters['executer_filters'][eid] = False

        for id in filters['strategy_filters'].keys():
            if type(filters['strategy_filters'][id]) != bool:
                filters['strategy_filters'][id] = True

        for id in filters['code_filters'].keys():
            if type(filters['code_filters'][id]) != bool:
                filters['code_filters'][id] = True

        return filters

    def set_group_filters(self, grpid:str, filters:dict):
        if grpid not in self.__config__["groups"]:
            return False

        grpInfo = self.__config__["groups"][grpid]
        self.__check_cache__(grpid, grpInfo)

        realfilters = {
            "strategy_filters":{},
            "code_filters":{},
            "executer_filters":{}
        }

        if "strategy_filters" in filters:
            for sid in filters["strategy_filters"]:
                if filters["strategy_filters"][sid]:
                    realfilters["strategy_filters"][sid] = {
                        "action":"redirect",
                        "target":0
                    }

        if "code_filters" in filters:
            for sid in filters["code_filters"]:
                if filters["code_filters"][sid]:
                    realfilters["code_filters"][sid] = {
                        "action":"redirect",
                        "target":0
                    }

        if "executer_filters" in filters:
            realfilters["executer_filters"] = filters["executer_filters"]
        
        filepath = os.path.join(grpInfo["path"], 'filters.json')
        backup_file(filepath)
        f = open(filepath, "w")
        f.write(json.dumps(realfilters, indent=4))
        f.close()
        return True
            