'''
Descripttion: 回测管理模块
version: 
Author: Wesley
Date: 2021-08-11 14:03:33
LastEditors: Wesley
LastEditTime: 2021-09-02 14:18:50
'''
import os
import json
import subprocess
import platform
import sys
import psutil
import hashlib
import datetime
import shutil
import json
import threading
import time

from wtpy import WtDtServo
from .WtLogger import WtLogger
from .EventReceiver import BtEventReceiver, BtEventSink

def isWindows():
    if "windows" in platform.system().lower():
        return True

    return False

def md5_str(v:str) -> str:
    return hashlib.md5(v.encode()).hexdigest()

def gen_btid(user:str, straid:str) -> str:
    now = datetime.datetime.now()
    s = user + "_" + straid + "_" + str(now.timestamp())
    return md5_str(s)

def gen_straid(user:str) -> str:
    now = datetime.datetime.now()
    s = user + "_" + str(now.timestamp())
    return md5_str(s)

class BtTaskSink:

    def __init__(self):
        pass

    def on_start(self, user:str, straid:str, btid:str):
        pass

    def on_stop(self, user:str, straid:str, btid:str):
        pass

    def on_state(self, user:str, straid:str, btid:str, statInfo:dict):
        pass

    def on_fund(self, user:str, straid:str, btid:str, fundInfo:dict):
        pass

class WtBtTask(BtEventSink):
    '''
    回测任务类
    '''
    def __init__(self, user:str, straid:str, btid:str, folder:str, logger:WtLogger = None, sink:BtTaskSink = None):
        self.user = user
        self.straid = straid
        self.btid = btid
        self.logger = logger
        self.folder = folder
        self.sink = sink
        
        self._cmd_line = None
        self._mq_url = "ipc:///wtpy/bt_%s.ipc" % (btid)
        self._ticks = 0
        self._state = 0
        self._procid = None
        self._evt_receiver = None

    def __check__(self):
         while True:
            time.sleep(1)
            pids = psutil.pids()
            if psutil.pid_exists(self._procid):
                continue
            else:
                print("%s process %d finished" % (self.btid, self._procid))
                if self.sink is not None:
                    self.sink.on_stop(self.user, self.straid, self.btid)
                break

    def run(self):
        if self._state != 0:
            return

        self._evt_receiver = BtEventReceiver(url=self._mq_url, logger=self.logger, sink=self)
        self._evt_receiver.run()
        self.logger.info("回测%s开始接收%s的通知信息" % (self.btid, self._mq_url))

        try:
            fullPath = os.path.join(self.folder, "runBT.py")
            if isWindows():
                self._procid = subprocess.Popen([sys.executable, fullPath],  # 需要执行的文件路径
                                cwd=self.folder, creationflags=subprocess.CREATE_NEW_CONSOLE).pid
            else:
                self._procid = subprocess.Popen([sys.executable, fullPath],  # 需要执行的文件路径
                                cwd=self.folder).pid

            self._cmd_line = sys.executable + " " + fullPath
        except:
            self.logger.info("回测%s启动异常" % (self.btid))

        self._state = 1

        self.logger.info("回测%s的已启动，进程ID: %d" % (self.btid, self._procid))

        self.watcher = threading.Thread(target=self.__check__, name=self.btid, daemon=True)
        self.watcher.start()

    @property
    def cmd_line(self) -> str:
        fullPath = os.path.join(self.folder, "runBT.py")
        if self._cmd_line is None:
            self._cmd_line = sys.executable + " " + fullPath
        return self._cmd_line

    def is_running(self, pids) -> bool:
        bNeedCheck = (self._procid is None) or (not psutil.pid_exists(self._procid))
        if bNeedCheck:
            for pid in pids:
                try:
                    pInfo = psutil.Process(pid)
                    cmdLine = pInfo.cmdline()
                    if len(cmdLine) == 0:
                        continue
                    # print(cmdLine)
                    cmdLine = ' '.join(cmdLine)
                    if self.cmd_line.upper() == cmdLine.upper():
                        self._procid = pid
                        self.logger.info("回测%s挂载成功，进程ID: %d" % (self.btid, self._procid))

                        if self._mq_url != '':
                            self._evt_receiver = BtEventReceiver(url=self._mq_url, logger=self.logger, sink=self)
                            self._evt_receiver.run()
                            self.logger.info("回测%s开始接收%s的通知信息" % (self.btid, self._mq_url))

                        self.watcher = threading.Thread(target=self.__check__, name=self.btid, daemon=True)
                        self.watcher.run()
                except:
                    pass
            return False

        return True

    def on_begin(self):
        if self.sink is not None:
            self.sink.on_start(self.user, self.straid, self.btid)

    def on_finish(self):
        pass

    def on_state(self, statInfo:dict):
        if self.sink is not None:
            self.sink.on_state(self.user, self.straid, self.btid, statInfo)
        print(statInfo)

    def on_fund(self, fundInfo:dict):
        if self.sink is not None:
            self.sink.on_fund(self.user, self.straid, self.btid, fundInfo)
        print(fundInfo)


class WtBtMon(BtTaskSink):
    '''
    回测管理器
    '''
    def __init__(self, deploy_folder:str, dtServo:WtDtServo = None, logger:WtLogger = None):
        self.path = deploy_folder
        self.user_stras = dict()
        self.user_bts = dict()
        self.logger = logger
        self.dt_servo = dtServo

        self.task_infos = dict()
        self.task_map = dict()

        self.__load_tasks__()

    def __load_user_data__(self, user:str):
        folder = os.path.join(self.path, user)
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, "marker.json")
        if not os.path.exists(filepath):
            return False

        f = open(filepath, "r")
        content = f.read()
        f.close()

        obj = json.loads(content)
        self.user_stras[user] = obj["strategies"]
        self.user_bts[user] = obj["backtests"]
        return True

    def __save_user_data__(self, user):
        folder = os.path.join(self.path, user)
        if not os.path.exists(folder):
            os.mkdir(folder)

        obj = {
            "strategies":{},
            "backtests":{}
        }

        if user in self.user_stras:
            obj["strategies"] = self.user_stras[user]

        if user in self.user_bts:
            obj["backtests"] = self.user_bts[user]

        filepath = os.path.join(folder, "marker.json")
        f = open(filepath, "w")
        f.write(json.dumps(obj, indent=4, ensure_ascii=False))
        f.close()
        return True

    def get_strategies(self, user:str) -> list:
        if user not in self.user_stras:
            bSucc = self.__load_user_data__(user)
        
            if not bSucc:
                return None

        ay = list()
        for straid in self.user_stras[user]:
            ay.append(self.user_stras[user][straid])
        return ay

    def add_strategy(self, user:str, name:str) -> dict:
        if user not in self.user_stras:
            self.__load_user_data__(user)

        if user not in self.user_stras:
            self.user_stras[user] = dict()

        straid = gen_straid(user)
        self.user_stras[user][straid] = {
            "id":straid,
            "name":name,
            "perform":{
                "days": 0,
                "total_return": 0,
                "annual_return": 0,
                "win_rate": 0,
                "max_falldown": 0,
                "max_profratio": 0,
                "std": 0,
                "down_std": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0
            }
        }

        folder = os.path.join(self.path, user, straid)
        if not os.path.exists(folder):
            os.mkdir(folder)

        fname = os.path.join(folder, "MyStrategy.py")
        srcfname = os.path.join(self.path, "template/MyStrategy.py")
        shutil.copyfile(srcfname, fname)

        self.__save_user_data__(user)

        return self.user_stras[user][straid]

    def del_strategy(self, user:str, straid:str):
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return False

        if straid not in self.user_stras[user]:
            return True

        folder = os.path.join(self.path, user, straid)
        if not os.path.exists(folder):
            return True

        delFolder = os.path.join(self.path, user, ".del")
        if not os.path.exists(delFolder):
            os.mkdir(delFolder)
        shutil.move(folder, delFolder)
        self.user_stras[user].pop(straid)
        self.__save_user_data__(user)
        return True
    
    def has_strategy(self, user:str, straid:str, btid:str = None) -> bool:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return False

        if btid is None:
            return straid in self.user_stras[user]
        else:
            return btid in self.user_bts[user]

    def get_strategy_code(self, user:str, straid:str, btid:str = None) -> str:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        if btid is None:
            path = os.path.join(self.path, user, straid, "MyStrategy.py")
            if not os.path.exists(path):
                return None

            f = open(path, "r", encoding="UTF-8")
            content = f.read()
            f.close()
            return content
        else:
            thisBts = self.user_bts[user]
            if btid not in thisBts:
                return None

            bt_path = os.path.join(self.path, "%s/%s/backtests/%s/runBT.py" % (user, straid, btid))
            f = open(bt_path, "r")
            content = f.read()
            f.close()
            return content

    def set_strategy_code(self, user:str, straid:str, content:str) -> bool:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return False

        path = os.path.join(self.path, user, straid, "MyStrategy.py")
        if not os.path.exists(path):
            return None

        f = open(path, "w", encoding="UTF-8")
        f.write(content)
        f.close()
        return True

    def get_backtests(self, user:str, straid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        if user not in self.user_bts:
            return None

        ay = list()
        for btid in self.user_bts[user]:
            ay.append(self.user_bts[user][btid])

        return ay

    def del_backtest(self, user:str, btid:str):
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return

        if user not in self.user_bts:
            return

        if btid in self.user_bts[user]:
            self.user_bts[user].pop(btid)
            self.logger.info("Backtest %s of %s deleted" % (btid, user))

            self.__save_user_data__(user)

    def get_bt_funds(self, user:str, straid:str, btid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/funds.csv" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        lines = lines[1:]

        funds = list()
        for line in lines:
            cells = line.split(",")
            if len(cells) > 10:
                continue

            tItem = {
                "date": int(cells[0]),
                "closeprofit": float(cells[1]),
                "dynprofit": float(cells[2]),
                "dynbalance": float(cells[3]),
                "fee": 0
            }

            if len(cells) > 4:
                tItem["fee"] = float(cells[4])

            funds.append(tItem)
        
        return funds

    def get_bt_trades(self, user:str, straid:str, btid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/trades.csv" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        lines = lines[1:]

        items = list()
        for line in lines:
            cells = line.split(",")
            if len(cells) > 10:
                continue

            item = {
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
                item["fee"] = float(cells[7])

            if len(cells) > 4:
                item["fee"] = float(cells[4])

            items.append(item)
        
        return items

    def get_bt_rounds(self, user:str, straid:str, btid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/closes.csv" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        lines = lines[1:]

        items = list()
        for line in lines:
            cells = line.split(",")

            item = {
                "code": cells[0],
                "direct": cells[1],
                "opentime": int(cells[2]),
                "openprice": float(cells[3]),
                "closetime": int(cells[4]),
                "closeprice": float(cells[5]),
                "qty": float(cells[6]),
                "profit": float(cells[7]),
                "maxprofit": float(cells[8]),
                "maxloss": float(cells[9]),
                "entertag": cells[11],
                "exittag": cells[12]
            }

            items.append(item)
        
        return items

    def get_bt_signals(self, user:str, straid:str, btid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/signals.csv" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        lines = lines[1:]

        items = list()
        for line in lines:
            cells = line.split(",")
            if len(cells) > 10:
                continue

            item = {
                "code": cells[0],
                "target": float(cells[1]),
                "sigprice": float(cells[2]),
                "gentime": cells[3],
                "tag": cells[4]
            }

            items.append(item)
        
        return items

    def get_bt_summary(self, user:str, straid:str, btid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/summary.json" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        content = f.read()
        f.close()

        obj = json.loads(content)
        return obj

    def get_bt_state(self, user:str, straid:str, btid:str) -> list:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/btenv.json" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        content = f.read()
        f.close()

        obj = json.loads(content)
        return obj

    def get_bt_state(self, user:str, straid:str, btid:str) -> dict:
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        filename = "%s/%s/backtests/%s/outputs_bt/%s/btenv.json" % (user, straid, btid, btid)
        filename = os.path.join(self.path, filename)
        if not os.path.exists(filename):
            return None

        f = open(filename, "r")
        content = f.read()
        f.close()

        thisBts[btid]["state"] = json.loads(content)

        return thisBts[btid]["state"]

    def update_bt_state(self, user:str, straid:str, btid:str, stateObj:dict):
        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None

        thisBts = self.user_bts[user]
        if btid not in thisBts:
            return None

        thisBts[btid]["state"] = stateObj

    def get_bt_kline(self, user:str, straid:str, btid:str) -> list:
        if self.dt_servo is None:
            return None

        if user not in self.user_bts:
            bSucc = self.__load_user_data__(user)

            if not bSucc:
                return None
        
        btState = self.get_bt_state(user, straid, btid)
        if btState is None:
            return None

        thisBts = self.user_bts[user]
        if "kline" not in thisBts[btid]:
            code = btState["code"]
            period = btState["period"]
            stime = btState["stime"]
            etime = btState["etime"]
            barList = self.dt_servo.get_bars(stdCode=code, period=period, fromTime=stime, endTime=etime)
            if barList is None:
                return None

            bars = list()
            for realBar in barList:
                bar = dict()
                if period[0] == 'd':
                    bar["time"] = realBar.date
                else:
                    bar["time"] = 1990*100000000 + realBar.time
                    bar["bartime"] = bar["time"]
                    bar["open"] = realBar.open
                    bar["high"] = realBar.high
                    bar["low"] = realBar.low
                    bar["close"] = realBar.close
                    bar["volume"] = realBar.vol
                bars.append(bar)
            thisBts[btid]["kline"] = bars

        return thisBts[btid]["kline"]

    def run_backtest(self, user:str, straid:str, fromTime:int, endTime:int, capital:float, slippage:int=0) -> dict:
        if user not in self.user_bts:
            self.__load_user_data__(user)

        if user not in self.user_bts:
            self.user_bts[user] = dict()
            
        btid = gen_btid(user, straid)

        # 生成回测目录
        folder = os.path.join(self.path, user, straid, "backtests")
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        folder = os.path.join(folder, btid)
        os.mkdir(folder)

        # 将策略文件复制到该目录下
        old_path = os.path.join(self.path, user, straid, "MyStrategy.py")
        new_path = os.path.join(folder, "MyStrategy.py")
        shutil.copyfile(old_path, new_path)

        # 初始化目录下的配置文件
        old_path = os.path.join(self.path, "template/configbt.json")
        new_path = os.path.join(folder, "configbt.json")
        f = open(old_path, "r", encoding="UTF-8")
        content = f.read()
        f.close()
        content = content.replace("$BTID$", btid)

        f = open(new_path, "w", encoding="UTF-8")
        f.write(content)
        f.close()

        old_path = os.path.join(self.path, "template/logcfgbt.json")
        new_path = os.path.join(folder, "logcfgbt.json")
        shutil.copyfile(old_path, new_path)

        old_path = os.path.join(self.path, "template/fees.json")
        new_path = os.path.join(folder, "fees.json")
        shutil.copyfile(old_path, new_path)

        old_path = os.path.join(self.path, "template/runBT.py")
        new_path = os.path.join(folder, "runBT.py")

        f = open(old_path, "r", encoding="UTF-8")
        content = f.read()
        f.close()
        content = content.replace("$FROMTIME$", str(fromTime))
        content = content.replace("$ENDTIME$", str(endTime))
        content = content.replace("$STRAID$", btid)
        content = content.replace("$CAPITAL$", str(capital))
        content = content.replace("$SLIPPAGE$", str(slippage))

        f = open(new_path, "w", encoding="UTF-8")
        f.write(content)
        f.close()

        btInfo = {
            "id":btid,
            "capital":capital,
            "runtime":datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"),
            "state":{
                "code": "",
                "period": "",
                "stime": fromTime,
                "etime": endTime,
                "progress": 0,
                "elapse": 0
            },
            "perform":{
                "days": 0,
                "total_return": 0,
                "annual_return": 0,
                "win_rate": 0,
                "max_falldown": 0,
                "max_profratio": 0,
                "std": 0,
                "down_std": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0
            }
        }

        self.user_bts[user][btid] = btInfo

        self.__save_user_data__(user)

        # 添加
        btTask = WtBtTask(user, straid, btid, folder, self.logger, sink=self)
        btTask.run()

        self.task_map[btid] = btTask

        # 这里还需要记录一下回测的任务，不然如果重启就恢复不了了
        taskInfo = {
            "user":user,
            "straid":straid,
            "btid":btid,
            "folder":folder
        }
        self.task_infos[btid]= taskInfo
        self.__save_tasks__()

        return btInfo

    def __update_bt_result__(self, user:str, straid:str, btid:str):
        if user not in self.user_bts:
            self.__load_user_data__(user)

        if user not in self.user_bts:
            self.user_bts[user] = dict()

        # 更新回测状态
        stateObj = self.get_bt_state(user, straid, btid)
        self.user_bts[user][btid]["state"] = stateObj

        # 更新回测结果摘要
        summaryObj = self.get_bt_summary(user, straid, btid)
        self.user_bts[user][btid]["perform"] = summaryObj
        self.user_stras[user][straid]["perform"] = summaryObj

        self.__save_user_data__(user)
    
    def __save_tasks__(self):
        obj = self.task_infos

        filename = os.path.join(self.path, "tasks.json")
        f = open(filename, "w")
        f.write(json.dumps(obj, indent=4))
        f.close()

    def __load_tasks__(self):
        filename = os.path.join(self.path, "tasks.json")
        if not os.path.exists(filename):
            return

        f = open(filename, "r")
        content = f.read()
        f.close()

        task_infos = json.loads(content)
        pids = psutil.pids()
        for btid in task_infos:
            tInfo = task_infos[btid].copy()
            tInfo["logger"] = self.logger
            btTask = WtBtTask(**tInfo)

            if btTask.is_running(pids):
                self.task_map[btid] = btTask
                self.task_infos[btid] = task_infos[btid]
                self.logger.info("回测任务%s已恢复")
            else:
                # 之前记录过测回测任务，执行完成了，要更新回测数据
                self.__update_bt_result__(tInfo["user"], tInfo["straid"], btid)
        
        self.__save_tasks__()
            

    def on_start(self, user:str, straid:str, btid:str):
        pass

    def on_stop(self, user:str, straid:str, btid:str):
        self.__update_bt_result__(user, straid, btid)

    def on_state(self, user:str, straid:str, btid:str, statInfo:dict):
        self.user_bts[user][btid]["state"] = statInfo

    def on_fund(self, user:str, straid:str, btid:str, fundInfo:dict):
        pass