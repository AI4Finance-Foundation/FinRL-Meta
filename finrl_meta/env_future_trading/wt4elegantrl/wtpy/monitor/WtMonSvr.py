from flask import Flask, session, redirect, request, make_response
from flask_compress  import Compress
import json
import datetime
import os
import hashlib
import sys
import base64
import chardet
import pytz

from .WtLogger import WtLogger
from .DataMgr import DataMgr, backup_file
from .PushSvr import PushServer
from .WatchDog import WatchDog, WatcherSink
from .EventReceiver import EventReceiver, EventSink
from .WtBtMon import WtBtMon
from wtpy import WtDtServo

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

#获取文件最后N行的函数
def get_tail(filename, N:int = 100, encoding="GBK") :
    filesize = os.path.getsize(filename)
    blocksize = 10240
    dat_file = open(filename, 'r', encoding=encoding)
    last_line = ""
    if filesize > blocksize :
        maxseekpoint = (filesize // blocksize)
        dat_file.seek((maxseekpoint-1)*blocksize)
    elif filesize :
        dat_file.seek(0, 0)
    lines =  dat_file.readlines()
    if lines :
        last_line = lines[-N:]
    dat_file.close()
    return ''.join(last_line), len(last_line)

def check_auth():
    usrInfo = session.get("userinfo")
    # session里没有用户信息
    if usrInfo is None:
        
        return False, {
            "result":-999,
            "message":"请先登录"
        }

    # session里有用户信息，则要读取
    exptime = session.get("expiretime")
    now = datetime.datetime.now().replace(tzinfo=pytz.timezone('UTC')).strftime("%Y.%m.%d %H:%M:%S")
    if now > exptime:
        return False, {
            "result":-999,
            "message":"登录已超时，请重新登录"
        }

    return True, usrInfo

def get_cfg_tree(root:str, name:str):
    if not os.path.exists(root):
        return {
            "label":name,
            "path":root,
            "exist":False,
            "isfile":False,
            "children":[]
        }

    if os.path.isfile(root):
        return {
            "label":name,
            "path":root,
            "exist":False,
            "isfile":True
        }

    ret = {
        "label":name,
        "path":root,
        "exist":True,
        "isfile":False,
        "children":[]
    }

    filepath = os.path.join(root, "run.py")
    ret['children'].append({
        "label":"run.py",
        "path":filepath,
        "exist":True,
        "isfile":True,
        "children":[]
    })

    filepath = os.path.join(root, "config.json")
    ret['children'].append({
        "label":"config.json",
        "path":filepath,
        "exist":True,
        "isfile":True,
        "children":[]
    })

    f = open(filepath, "r")
    content = f.read()
    f.close()
    cfgObj = json.loads(content)
    if "executers" in cfgObj:
        filename = cfgObj["executers"]
        if type(filename) == str:
            filepath = os.path.join(root, filename)
            ret['children'].append({
                "label":filename,
                "path":filepath,
                "exist":True,
                "isfile":True,
                "children":[]
            })

    if "parsers" in cfgObj:
        filename = cfgObj["parsers"]
        if type(filename) == str:
            filepath = os.path.join(root, filename)
            ret['children'].append({
                "label":filename,
                "path":filepath,
                "exist":True,
                "isfile":True,
                "children":[]
            })

    if "traders" in cfgObj:
        filename = cfgObj["traders"]
        if type(filename) == str:
            filepath = os.path.join(root, filename)
            ret['children'].append({
                "label":filename,
                "path":filepath,
                "exist":True,
                "isfile":True,
                "children":[]
            })

    filepath = os.path.join(root, 'generated')
    ret["children"].append(get_path_tree(filepath, 'generated', True))
        
    return ret

def get_path_tree(root:str, name:str, hasFile:bool = True):
    if not os.path.exists(root):
        return {
            "label":name,
            "path":root,
            "exist":False,
            "isfile":False,
            "children":[]
        }

    if os.path.isfile(root):
        return {
            "label":name,
            "path":root,
            "exist":False,
            "isfile":True
        }

    ret = {
        "label":name,
        "path":root,
        "exist":True,
        "isfile":False,
        "children":[]
    }
    files = os.listdir(root, )
    for filename in files:
        if filename in ['__pycache__', '.vscode', 'wtpy', '__init__.py']:
            continue
        if filename[-3:] == 'pyc':
            continue

        filepath = os.path.join(root, filename)
        if os.path.isfile(filepath):
            if not hasFile:
                continue
            else:
                ret["children"].append({
                    "label":filename,
                    "path":filepath,
                    "exist":True,
                    "isfile":True})
        else:
            ret["children"].append(get_path_tree(filepath, filename, hasFile))

        ay1 = list()
        ay2 = list()
        for item in ret["children"]:
            if item["isfile"]:
                ay2.append(item)
            else:
                ay1.append(item)
        ay = ay1 + ay2
        ret["children"] = ay
    return ret

class WtMonSvr(WatcherSink):

    def __init__(self, static_folder:str="", static_url_path="/", deploy_dir="C:/"):
        if len(static_folder) == 0:
            static_folder = 'static'

        self.logger = WtLogger(__name__, "WtMonSvr.log")

        # 数据管理器，主要用于缓存各组合的数据
        self.__data_mgr__ = DataMgr('data.db', logger=self.logger)

        self.__bt_mon__:WtBtMon = None
        self.__dt_servo__:WtDtServo = None

        # 看门狗模块，主要用于调度各个组合启动关闭
        self._dog = WatchDog(sink=self, db=self.__data_mgr__.get_db(), logger=self.logger)

        app = Flask(__name__, instance_relative_config=True, static_folder=static_folder, static_url_path=static_url_path)
        app.secret_key = "!@#$%^&*()"
        Compress(app)
        # app.debug = True
        self.app = app
        self.worker = None
        self.deploy_dir = deploy_dir
        self.deploy_tree = None

        self.push_svr = PushServer(app, self.__data_mgr__, self.logger)

        self.init_mgr_apis(app)

    def set_bt_mon(self, btMon:WtBtMon):
        self.__bt_mon__ = btMon
        self.init_bt_apis(self.app)

    def set_dt_servo(self, dtServo:WtDtServo):
        self.__dt_servo__ = dtServo

    def init_bt_apis(self, app:Flask):

        # 拉取K线数据
        @app.route("/bt/qrybars", methods=["POST"])
        def qry_bt_bars():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            if self.__dt_servo__ is None:
                ret = {
                    "result":-2,
                    "message":"没有配置数据伺服"
                }
                return pack_rsp(ret)

            stdCode = get_param(json_data, "code")
            period = get_param(json_data, "period")
            fromTime = get_param(json_data, "stime", int, None)
            dataCount = get_param(json_data, "count", int, None)
            endTime = get_param(json_data, "etime", int)

            bars = self.__dt_servo__.get_bars(stdCode=stdCode, period=period, fromTime=fromTime, dataCount=dataCount, endTime=endTime)
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

        
        # 拉取用户策略列表
        @app.route("/bt/qrystras", methods=["POST"])
        def qry_my_stras():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            ret = {
                "result":0,
                "message":"OK",
                "strategies": self.__bt_mon__.get_strategies(user)
            }

            return pack_rsp(ret)

        # 拉取策略代码
        @app.route("/bt/qrycode", methods=["POST"])
        def qry_stra_code():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略代码不存在"
                    }
                else:
                    content = self.__bt_mon__.get_strategy_code(user, straid)
                    ret = {
                        "result":0,
                        "message":"OK",
                        "content":content
                    }

            return pack_rsp(ret)

        # 提交策略代码
        @app.route("/bt/setcode", methods=["POST"])
        def set_stra_code():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            content = get_param(json_data, "content")
            if len(content) == 0 or len(straid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID和代码不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = self.__bt_mon__.set_strategy_code(user, straid, content)
                    if ret:
                        ret = {
                            "result":0,
                            "message":"OK"
                        }
                    else:
                        ret = {
                            "result":-3,
                            "message":"保存策略代码失败"
                        }

            return pack_rsp(ret)

        # 添加用户策略
        @app.route("/bt/addstra", methods=["POST"])
        def cmd_add_stra():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            name = get_param(json_data, "name")
            if len(name) == 0:
                ret = {
                    "result":-2,
                    "message":"策略名称不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-3,
                    "message":"回测管理器未配置"
                }
                return pack_rsp(ret)

            straInfo = self.__bt_mon__.add_strategy(user, name)
            if straInfo is None:
                ret = {
                    "result":-4,
                    "message":"策略添加失败"
                }
            else:
                ret = {
                    "result":0,
                    "message":"OK",
                    "strategy": straInfo
                }

            return pack_rsp(ret)

        # 删除用户策略
        @app.route("/bt/delstra", methods=["POST"])
        def cmd_del_stra():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            if len(straid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = self.__bt_mon__.del_strategy(user, straid)
                    if ret:
                        ret = {
                            "result":0,
                            "message":"OK"
                        }
                    else:
                        ret = {
                            "result":-3,
                            "message":"保存策略代码失败"
                        }

            return pack_rsp(ret)

        # 获取策略回测列表
        @app.route("/bt/qrystrabts", methods=["POST"])
        def qry_stra_bts():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            if len(straid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = {
                        "result":0,
                        "message":"OK",
                        "backtests":self.__bt_mon__.get_backtests(user, straid)
                    }

            return pack_rsp(ret)

        # 获取策略回测信号
        @app.route("/bt/qrybtsigs", methods=["POST"])
        def qry_stra_bt_signals():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            btid = get_param(json_data, "btid")
            if len(straid) == 0 or len(btid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID和回测ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = {
                        "result":0,
                        "message":"OK",
                        "signals":self.__bt_mon__.get_bt_signals(user, straid, btid)
                    }

            return pack_rsp(ret)

        # 删除策略回测列表
        @app.route("/bt/delstrabt", methods=["POST"])
        def cmd_del_stra_bt():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            btid = get_param(json_data, "btid")
            if len(btid) == 0:
                ret = {
                    "result":-2,
                    "message":"回测ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                self.__bt_mon__.del_backtest(user, btid)
                ret = {
                    "result":0,
                    "message":"OK"
                }

            return pack_rsp(ret)

        # 获取策略回测成交
        @app.route("/bt/qrybttrds", methods=["POST"])
        def qry_stra_bt_trades():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            btid = get_param(json_data, "btid")
            if len(straid) == 0 or len(btid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID和回测ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = {
                        "result":0,
                        "message":"OK",
                        "trades":self.__bt_mon__.get_bt_trades(user, straid, btid)
                    }

            return pack_rsp(ret)

        # 获取策略回测资金
        @app.route("/bt/qrybtfunds", methods=["POST"])
        def qry_stra_bt_funds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            btid = get_param(json_data, "btid")
            if len(straid) == 0 or len(btid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID和回测ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = {
                        "result":0,
                        "message":"OK",
                        "funds":self.__bt_mon__.get_bt_funds(user, straid, btid)
                    }

            return pack_rsp(ret)

        # 获取策略回测回合
        @app.route("/bt/qrybtrnds", methods=["POST"])
        def qry_stra_bt_rounds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            straid = get_param(json_data, "straid")
            btid = get_param(json_data, "btid")
            if len(straid) == 0 or len(btid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID和回测ID不能为空"
                }
                return pack_rsp(ret)

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    ret = {
                        "result":0,
                        "message":"OK",
                        "rounds":self.__bt_mon__.get_bt_rounds(user, straid, btid)
                    }

            return pack_rsp(ret)

        # 启动策略回测
        @app.route("/bt/runstrabt", methods=["POST"])
        def cmd_run_stra_bt():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, userInfo = check_auth()
            if not bSucc:
                return pack_rsp(userInfo)

            user = userInfo["loginid"]
            role = userInfo["role"]
            if role not in ['researcher','superman']:
                ret = {
                    "result":-1,
                    "message":"没有权限"
                }
                return pack_rsp(ret)

            curDt = int(datetime.datetime.now().strftime("%Y%m%d"))

            straid = get_param(json_data, "straid")
            fromtime = get_param(json_data, "stime", int, defVal=curDt)
            endtime = get_param(json_data, "etime", int, defVal=curDt)
            capital = get_param(json_data, "capital", float, defVal=500000)
            slippage = get_param(json_data, "slippage", int, defVal=0)
            if len(straid) == 0:
                ret = {
                    "result":-2,
                    "message":"策略ID不能为空"
                }
                return pack_rsp(ret)

            if fromtime > endtime:
                fromtime,endtime = endtime,fromtime

            fromtime = fromtime*10000 + 900
            endtime = endtime*10000 + 1515

            if self.__bt_mon__ is None:
                ret = {
                    "result":-1,
                    "message":"回测管理器未配置"
                }
            else:
                if not self.__bt_mon__.has_strategy(user, straid):
                    ret = {
                        "result":-2,
                        "message":"策略不存在"
                    }
                else:
                    btInfo = self.__bt_mon__.run_backtest(user,straid,fromtime,endtime,capital,slippage)
                    ret = {
                        "result":0,
                        "message":"OK",
                        "backtest": btInfo
                    }

            return pack_rsp(ret)

    def init_mgr_apis(self, app:Flask):

        @app.route("/console", methods=["GET"])
        def stc_console_index():
            return redirect("./console/index.html")

        @app.route("/mobile", methods=["GET"])
        def stc_mobile_index():
            return redirect("./mobile/index.html")


        '''下面是API接口的编写'''
        @app.route("/mgr/login", methods=["POST"])
        def cmd_login():
            
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            if True:
                user = get_param(json_data, "loginid")
                pwd = get_param(json_data, "passwd")

                if len(user) == 0 or len(pwd) == 0:
                    ret = {
                        "result":-1,
                        "message":"用户名和密码不能为空"
                    }
                else:
                    encpwd = hashlib.md5((user+pwd).encode("utf-8")).hexdigest()
                    now = datetime.datetime.now()
                    usrInf = self.__data_mgr__.get_user(user)
                    if usrInf is None:
                        ret = {
                            "result":-1,
                            "message":"用户不存在"
                        }
                    elif encpwd != usrInf["passwd"]:
                        ret = {
                            "result":-1,
                            "message":"登录密码错误"
                        }
                    else:
                        usrInf.pop("passwd")
                        usrInf["loginip"]=request.remote_addr
                        usrInf["logintime"]=now.strftime("%Y/%m/%d %H:%M:%S")

                        exptime = now + datetime.timedelta(minutes=360)  #360分钟令牌超时
                        session["userinfo"] = usrInf
                        session["expiretime"] = exptime.replace(tzinfo=pytz.timezone('UTC')).strftime("%Y.%m.%d %H:%M:%S")

                        ret = {
                            "result":0,
                            "message":"Ok",
                            "userinfo":usrInf
                        }

                        self.__data_mgr__.log_action(usrInf, "login", json.dumps(request.headers.get('User-Agent')))
            else:
                ret = {
                    "result":-1,
                    "message":"请求处理出现异常",
                }
                if session.get("userinfo") is not None:
                    session.pop("userinfo")

            return pack_rsp(ret)

        # 修改密码
        @app.route("/mgr/modpwd", methods=["POST"])
        def mod_pwd():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            oldpwd = get_param(json_data, "oldpwd")
            newpwd = get_param(json_data, "newpwd")

            if len(oldpwd) == 0 or len(newpwd) == 0:
                ret = {
                    "result":-1,
                    "message":"新旧密码都不能为空"
                }
            else:
                user = adminInfo["loginid"]
                oldencpwd = hashlib.md5((user+oldpwd).encode("utf-8")).hexdigest()
                usrInf = self.__data_mgr__.get_user(user)
                if usrInf is None:
                    ret = {
                        "result":-1,
                        "message":"用户不存在"
                    }
                else:
                    if oldencpwd != usrInf["passwd"]:
                        ret = {
                            "result":-1,
                            "message":"旧密码错误"
                        }
                    else:
                        if 'builtin' in usrInf and usrInf["builtin"]:
                            #如果是内建账号要改密码，则先添加用户
                            usrInf["passwd"] = oldpwd
                            self.__data_mgr__.add_user(usrInf, user)
                            print("%s是内建账户，自动添加到数据库中" % user)

                        newencpwd = hashlib.md5((user+newpwd).encode("utf-8")).hexdigest()
                        self.__data_mgr__.mod_user_pwd(user, newencpwd, user)

                        ret = {
                            "result":0,
                            "message":"Ok"
                        }

            return pack_rsp(ret)

        # 添加组合
        @app.route("/mgr/addgrp", methods=["POST"])
        def cmd_add_group():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            id = get_param(json_data, "groupid")
            name = get_param(json_data, "name")
            path = get_param(json_data, "path")
            info = get_param(json_data, "info")
            gtype = get_param(json_data, "gtype")
            env = get_param(json_data, "env")
            datmod = get_param(json_data, "datmod")
            mqurl = get_param(json_data, "mqurl")

            action = get_param(json_data, "action")
            if action == "":
                action = "add"

            if len(id) == 0 or len(name) == 0 or len(gtype) == 0:
                ret = {
                    "result":-1,
                    "message":"组合ID、名称、类型都不能为空"
                }
            elif not os.path.exists(path) or not os.path.isdir(path):
                ret = {
                    "result":-2,
                    "message":"组合运行目录不正确"
                }
            elif action == "add" and self.__data_mgr__.has_group(id):
                ret = {
                    "result":-3,
                    "message":"组合ID不能重复"
                }
            else:
                try:
                    grpInfo = {
                        "id":id,
                        "name":name,
                        "path":path,
                        "info":info,
                        "gtype":gtype,
                        "datmod":datmod,
                        "env":env,
                        "mqurl":mqurl
                    }   

                    if self.__data_mgr__.add_group(grpInfo):
                        ret = {
                            "result":0,
                            "message":"Ok"
                        }

                        if action == "add":
                            self.__data_mgr__.log_action(adminInfo, "addgrp", json.dumps(grpInfo))
                        else:
                            self.__data_mgr__.log_action(adminInfo, "modgrp", json.dumps(grpInfo))

                        self._dog.updateMQURL(id, mqurl)
                    else:
                        ret = {
                            "result":-2,
                            "message":"添加用户失败"
                        }
                except:
                    ret = {
                        "result":-1,
                        "message":"请求解析失败"
                    }

            return pack_rsp(ret)

        # 删除组合
        @app.route("/mgr/delgrp", methods=["POST"])
        def cmd_del_group():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            id = get_param(json_data, "groupid")

            if len(id) == 0:
                ret = {
                    "result":-1,
                    "message":"组合ID不能为空"
                }
            elif not self.__data_mgr__.has_group(id):
                ret = {
                    "result":-3,
                    "message":"该组合不存在"
                }
            elif self._dog.isRunning(id):
                ret = {
                    "result":-3,
                    "message":"请先停止该组合"
                }
            else:
                if True:
                    self._dog.delApp(id)
                    self.__data_mgr__.del_group(id)
                    ret = {
                        "result":0,
                        "message":"Ok"
                    }

                    self.__data_mgr__.log_action(adminInfo, "delgrp", id)
                else:
                    ret = {
                        "result":-1,
                        "message":"请求解析失败"
                    }

            return pack_rsp(ret)

        # 组合停止
        @app.route("/mgr/stopgrp", methods=["POST"])
        def cmd_stop_group():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)
            
            grpid = get_param(json_data, "groupid")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                if self._dog.isRunning(grpid):
                    self._dog.stop(grpid)
                ret = {
                    "result":0,
                    "message":"Ok"
                }

                self.__data_mgr__.log_action(adminInfo, "stopgrp", grpid)

            return pack_rsp(ret)
        
        # 组合启动
        @app.route("/mgr/startgrp", methods=["POST"])
        def cmd_start_group():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)
            
            grpid = get_param(json_data, "groupid")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                if not self._dog.isRunning(grpid):
                    self._dog.start(grpid)
                ret = {
                    "result":0,
                    "message":"Ok"
                }
                self.__data_mgr__.log_action(adminInfo, "startgrp", grpid)

            return pack_rsp(ret)

        # 获取执行的python进程的路径
        @app.route("/mgr/qryexec", methods=["POST"])
        def qry_exec_path():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            ret = {
                "result":0,
                "message":"Ok",
                "path": sys.executable
            }

            return pack_rsp(ret)

        # 配置监控
        @app.route("/mgr/qrymon", methods=["POST"])
        def qry_mon_cfg():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            grpid = get_param(json_data, "groupid")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                monCfg = self._dog.getAppConf(grpid)
                if monCfg is None:
                    ret = {
                        "result":0,
                        "message":"ok"
                    }
                else:
                    ret = {
                        "result":0,
                        "message":"ok",
                        "config":monCfg
                    }

            return pack_rsp(ret)

        # 配置监控
        @app.route("/mgr/cfgmon", methods=["POST"])
        def cmd_config_monitor():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            #这里本来是要做检查的，算了，先省事吧
            isGrp = get_param(json_data, "group", bool, False)
            
            self._dog.applyAppConf(json_data, isGrp)
            ret = {
                "result":0,
                "message":"ok"
            }
            self.__data_mgr__.log_action(adminInfo, "cfgmon", json.dumps(json_data))

            return pack_rsp(ret)

        # 查询目录结构
        @app.route("/mgr/qrydir", methods=["POST"])
        def qry_directories():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            if True:
                if self.deploy_tree is None:
                    self.deploy_tree = get_path_tree(self.deploy_dir, "root")

                ret = {
                    "result":0,
                    "message":"Ok",
                    "tree":self.deploy_tree
                }
            else:
                ret = {
                    "result":-1,
                    "message":"请求解析失败"
                }

            return pack_rsp(ret)

        # 查询目录结构
        @app.route("/mgr/qrygrpdir", methods=["POST"])
        def qry_grp_directories():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "groupid")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                monCfg = self.__data_mgr__.get_group(grpid)

                ret = {
                    "result":0,
                    "message":"Ok",
                    "tree": get_cfg_tree(monCfg["path"], "root")
                }

            return pack_rsp(ret)

        # 查询组合列表
        @app.route("/mgr/qrygrp", methods=["POST"])
        def qry_groups():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            try:
                groups = self.__data_mgr__.get_groups()
                for grpInfo in groups:
                    grpInfo["running"] = self._dog.isRunning(grpInfo["id"])
                ret = {
                    "result":0,
                    "message":"Ok",
                    "groups":groups
                }
            except:
                ret = {
                    "result":-1,
                    "message":"请求解析失败"
                }

            return pack_rsp(ret)

        # 查询文件信息
        @app.route("/mgr/qrygrpfile", methods=["POST"])
        def qry_group_file():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "groupid")
            path = get_param(json_data, "path")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                monCfg = self.__data_mgr__.get_group(grpid)
                root = monCfg["path"]
                if path[:len(root)] != root:
                    ret = {
                        "result":-1,
                        "message":"目标文件不在当前组合下"
                    }
                else:
                    f = open(path,'rb')
                    content = f.read()
                    f.close()

                    encoding = chardet.detect(content)["encoding"]
                    content = content.decode(encoding)

                    ret = {
                        "result":0,
                        "message":"Ok",
                        "content": content
                    }

            return pack_rsp(ret)

        # 提交组合文件
        @app.route("/mgr/cmtgrpfile", methods=["POST"])
        def cmd_commit_group_file():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "groupid")
            content = get_param(json_data, "content")
            path = get_param(json_data, "path")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                monCfg = self.__data_mgr__.get_group(grpid)
                root = monCfg["path"]
                if path[:len(root)] != root:
                    ret = {
                        "result":-1,
                        "message":"目标文件不在当前组合下"
                    }
                else:
                    try:
                        f = open(path,'rb')
                        old_content = f.read()
                        f.close()
                        encoding = chardet.detect(old_content)["encoding"]

                        backup_file(path)

                        f = open(path,'wb')
                        f.write(content.encode(encoding))
                        f.close()

                        ret = {
                            "result":0,
                            "message":"Ok"
                        }
                    except:
                        ret = {
                            "result":-1,
                            "message":"文件保存失败"
                        }

            return pack_rsp(ret)
        
        # 查询策略列表
        @app.route("/mgr/qrystras", methods=["POST"])
        def qry_strategys():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "groupid")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"Ok",
                    "strategies":self.__data_mgr__.get_strategies(grpid)
                }

            return pack_rsp(ret)

        # 查询通道列表
        @app.route("/mgr/qrychnls", methods=["POST"])
        def qry_channels():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "groupid")
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"Ok",
                    "channels":self.__data_mgr__.get_channels(grpid)
                }

            return pack_rsp(ret)

        # 查询组合日志
        @app.route("/mgr/qrylogs", methods=["POST"])
        def qry_logs():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "id")
            logtype = get_param(json_data, "type")

            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                grpInfo = self.__data_mgr__.get_group(grpid)
                try:
                    logfolder = os.path.join(grpInfo["path"], "./Logs/")
                    file_list = os.listdir(logfolder)
                    targets = list()
                    for fname in file_list:
                        if fname[:6] == "Runner":
                            targets.append(fname)

                    targets.sort()
                    filename = os.path.join(logfolder, targets[-1])
                    content,lines = get_tail(filename, 100)
                    ret = {
                        "result":0,
                        "message":"Ok",
                        "content":content,
                        "lines":lines
                    }
                except:
                    ret = {
                        "result":-1,
                        "message":"请求解析失败"
                    }

            return pack_rsp(ret)

        # 查询策略成交
        @app.route("/mgr/qrytrds", methods=["POST"])
        def qry_trades():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            sid = get_param(json_data, "strategyid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "trades": self.__data_mgr__.get_trades(gid, sid)
                }
                    

            return pack_rsp(ret)

        # 查询策略信号
        @app.route("/mgr/qrysigs", methods=["POST"])
        def qry_signals():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            sid = get_param(json_data, "strategyid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "signals": self.__data_mgr__.get_signals(gid, sid)
                }
                    

            return pack_rsp(ret)

        # 查询策略回合
        @app.route("/mgr/qryrnds", methods=["POST"])
        def qry_rounds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            sid = get_param(json_data, "strategyid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "rounds": self.__data_mgr__.get_rounds(gid, sid)
                }

            return pack_rsp(ret)

        # 查询策略持仓
        @app.route("/mgr/qrypos", methods=["POST"])
        def qry_positions():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            sid = get_param(json_data, "strategyid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "positions": self.__data_mgr__.get_positions(gid, sid)
                }

            return pack_rsp(ret)

        # 查询策略持仓
        @app.route("/mgr/qryfunds", methods=["POST"])
        def qry_funds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            sid = get_param(json_data, "strategyid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "funds": self.__data_mgr__.get_funds(gid, sid)
                }

            return pack_rsp(ret)

        # 查询通道订单
        @app.route("/mgr/qrychnlords", methods=["POST"])
        def qry_channel_orders():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            cid = get_param(json_data, "channelid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "orders": self.__data_mgr__.get_channel_orders(gid, cid)
                }

            return pack_rsp(ret)

        # 查询通道成交
        @app.route("/mgr/qrychnltrds", methods=["POST"])
        def qry_channel_trades():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            cid = get_param(json_data, "channelid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "trades": self.__data_mgr__.get_channel_trades(gid, cid)
                }

            return pack_rsp(ret)

        # 查询通道持仓
        @app.route("/mgr/qrychnlpos", methods=["POST"])
        def qry_channel_position():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            cid = get_param(json_data, "channelid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "positions": self.__data_mgr__.get_channel_positions(gid, cid)
                }

            return pack_rsp(ret)

        # 查询通道资金
        @app.route("/mgr/qrychnlfund", methods=["POST"])
        def qry_channel_funds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")
            cid = get_param(json_data, "channelid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "funds": self.__data_mgr__.get_channel_funds(gid, cid)
                }

            return pack_rsp(ret)

        # 查询用户列表
        @app.route("/mgr/qryusers", methods=["POST"])
        def qry_users():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            users = self.__data_mgr__.get_users()
            for usrInfo in users:
                usrInfo.pop("passwd")
            
            ret = {
                "result":0,
                "message":"",
                "users": users
            }
                

            return pack_rsp(ret)

        # 提交用户信息
        @app.route("/mgr/cmtuser", methods=["POST"])
        def cmd_commit_user():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            self.__data_mgr__.add_user(json_data, adminInfo["loginid"])
            ret = {
                "result":0,
                "message":"Ok"
            }

            self.__data_mgr__.log_action(adminInfo, "cmtuser", json.dumps(json_data))

            return pack_rsp(ret)

        # 删除用户
        @app.route("/mgr/deluser", methods=["POST"])
        def cmd_delete_user():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            loginid = get_param(json_data, "loginid")

            if self.__data_mgr__.del_user(loginid, adminInfo["loginid"]):
                self.__data_mgr__.log_action(adminInfo, "delusr", loginid)
            ret = {
                "result":0,
                "message":"Ok"
            }

            return pack_rsp(ret)

        # 修改密码
        @app.route("/mgr/resetpwd", methods=["POST"])
        def reset_pwd():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            user = get_param(json_data, "loginid")
            pwd = get_param(json_data, "passwd")

            if len(pwd) == 0 or len(user) == 0:
                ret = {
                    "result":-1,
                    "message":"密码都不能为空"
                }
            else:
                encpwd = hashlib.md5((user+pwd).encode("utf-8")).hexdigest()
                usrInf = self.__data_mgr__.get_user(user)
                if usrInf is None:
                    ret = {
                        "result":-1,
                        "message":"用户不存在"
                    }
                else:
                    self.__data_mgr__.mod_user_pwd(user, encpwd, adminInfo["loginid"])
                    self.__data_mgr__.log_action(adminInfo, "resetpwd", loginid)
                    ret = {
                        "result":0,
                        "message":"Ok"
                    }

            return pack_rsp(ret)

        # 查询操作记录
        @app.route("/mgr/qryacts", methods=["POST"])
        def qry_actions():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            sdate = get_param(json_data, "sdate")
            edate = get_param(json_data, "edate")

            ret = {
                "result":0,
                "message":"",
                "actions": self.__data_mgr__.get_actions(sdate, edate)
            }   

            return pack_rsp(ret)

        # 查询全部调度
        @app.route("/mgr/qrymons", methods=["POST"])
        def qry_mon_apps():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            schedules = self._dog.get_apps()
            for appid in schedules:
                schedules[appid]["group"] = self.__data_mgr__.has_group(appid)                

            ret = {
                "result":0,
                "message":"",
                "schedules": schedules
            }   

            return pack_rsp(ret)

        @app.route("/mgr/startapp", methods=["POST"])
        def cmd_start_app():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)
            
            appid = get_param(json_data, "appid")
            if not self._dog.has_app(appid):
                ret = {
                    "result":-1,
                    "message":"App不存在"
                }
            else:
                if not self._dog.isRunning(appid):
                    self._dog.start(appid)
                ret = {
                    "result":0,
                    "message":"Ok"
                }
                self.__data_mgr__.log_action(adminInfo, "startapp", appid)

            return pack_rsp(ret)

        # 组合停止
        @app.route("/mgr/stopapp", methods=["POST"])
        def cmd_stop_app():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)
            
            appid = get_param(json_data, "appid")
            if not self._dog.has_app(appid):
                ret = {
                    "result":-1,
                    "message":"App不存在"
                }
            else:
                if self._dog.isRunning(appid):
                    self._dog.stop(appid)
                ret = {
                    "result":0,
                    "message":"Ok"
                }

                self.__data_mgr__.log_action(adminInfo, "stopapp", appid)

            return pack_rsp(ret)

        # 查询调度日志
        @app.route("/mgr/qrymonlog", methods=["POST"])
        def qry_mon_logs():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)
            
            filename = os.getcwd() + "/logs/WtMonSvr.log"
            content,lines = get_tail(filename, 100, "UTF-8")
            ret = {
                "result":0,
                "message":"Ok",
                "content":content,
                "lines":lines
            }

            return pack_rsp(ret)

        # 删除调度任务
        @app.route("/mgr/delapp", methods=["POST"])
        def cmd_del_app():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, adminInfo = check_auth()
            if not bSucc:
                return pack_rsp(adminInfo)

            id = get_param(json_data, "appid")

            if len(id) == 0:
                ret = {
                    "result":-1,
                    "message":"组合ID不能为空"
                }
            elif self.__data_mgr__.has_group(id):
                ret = {
                    "result":-2,
                    "message":"该调度任务是策略组合，请从组合管理删除"
                }
            elif not self._dog.has_app(id):
                ret = {
                    "result":-3,
                    "message":"该调度任务不存在"
                }
            elif self._dog.isRunning(id):
                ret = {
                    "result":-4,
                    "message":"请先停止该任务"
                }
            else:
                if True:
                    self._dog.delApp(id)
                    ret = {
                        "result":0,
                        "message":"Ok"
                    }

                    self.__data_mgr__.log_action(adminInfo, "delapp", id)
                else:
                    ret = {
                        "result":-1,
                        "message":"请求解析失败"
                    }

            return pack_rsp(ret)

        # 查询组合持仓
        @app.route("/mgr/qryportpos", methods=["POST"])
        def qry_group_positions():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "positions": self.__data_mgr__.get_group_positions(gid)
                }
                    

            return pack_rsp(ret)

        # 查询组合成交
        @app.route("/mgr/qryporttrd", methods=["POST"])
        def qry_group_trades():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "trades": self.__data_mgr__.get_group_trades(gid)
                }
                    

            return pack_rsp(ret)

        # 查询组合回合
        @app.route("/mgr/qryportrnd", methods=["POST"])
        def qry_group_rounds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "rounds": self.__data_mgr__.get_group_rounds(gid)
                }
                    

            return pack_rsp(ret)
        
        # 查询组合资金
        @app.route("/mgr/qryportfunds", methods=["POST"])
        def qry_group_funds():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "funds": self.__data_mgr__.get_group_funds(gid)
                }
                    

            return pack_rsp(ret)

        # 查询组合绩效分析
        @app.route("/mgr/qryportperfs", methods=["POST"])
        def qry_group_perfs():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "performance": self.__data_mgr__.get_group_performances(gid)
                }
                    
            return pack_rsp(ret)

        # 查询组合过滤器
        @app.route("/mgr/qryportfilters", methods=["POST"])
        def qry_group_filters():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            gid = get_param(json_data, "groupid")

            if not self.__data_mgr__.has_group(gid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                ret = {
                    "result":0,
                    "message":"",
                    "filters": self.__data_mgr__.get_group_filters(gid)
                }
                    
            return pack_rsp(ret)

        # 提交组合过滤器
        @app.route("/mgr/cmtgrpfilters", methods=["POST"])
        def cmd_commit_group_filters():
            bSucc, json_data = parse_data()
            if not bSucc:
                return pack_rsp(json_data)

            bSucc, usrInfo = check_auth()
            if not bSucc:
                return pack_rsp(usrInfo)

            grpid = get_param(json_data, "groupid")
            filters = get_param(json_data, "filters", type=dict)
            if not self.__data_mgr__.has_group(grpid):
                ret = {
                    "result":-1,
                    "message":"组合不存在"
                }
            else:
                try:
                    self.__data_mgr__.set_group_filters(grpid, filters)
                    ret = {
                        "result":0,
                        "message":"Ok"
                    }
                except:
                    ret = {
                        "result":-1,
                        "message":"过滤器保存失败"
                    }

            return pack_rsp(ret)
            
    
    def __run_impl__(self, port:int, host:str):
        self._dog.run()
        self.push_svr.run(port = port, host = host)
    
    def run(self, port:int = 8080, host="0.0.0.0", bSync:bool = True):
        if bSync:
            self.__run_impl__(port, host)
        else:
            import threading
            self.worker = threading.Thread(target=self.__run_impl__, args=(port,host,))
            self.worker.setDaemon(True)
            self.worker.start()

    def init_logging(self):
        pass

    def on_start(self, grpid:str):
        if self.__data_mgr__.has_group(grpid):
            self.push_svr.notifyGrpEvt(grpid, 'start')

    def on_stop(self, grpid:str):
        if self.__data_mgr__.has_group(grpid):
            self.push_svr.notifyGrpEvt(grpid, 'stop')
    
    def on_output(self, grpid:str, tag:str, time:int, message:str):
        if self.__data_mgr__.has_group(grpid):
            self.push_svr.notifyGrpLog(grpid, tag, time, message)

    def on_order(self, grpid:str, chnl:str, ordInfo:dict):
        self.push_svr.notifyGrpChnlEvt(grpid, chnl, 'order', ordInfo)

    def on_trade(self, grpid:str, chnl:str, trdInfo:dict):
        self.push_svr.notifyGrpChnlEvt(grpid, chnl, 'trade', trdInfo)
    
    def on_notify(self, grpid:str, chnl:str, message:str):
        self.push_svr.notifyGrpChnlEvt(grpid, chnl, 'notify', message)