'''
Descripttion: Automatically generated file comment
version: 
Author: Wesley
Date: 2021-07-27 09:53:43
LastEditors: Wesley
LastEditTime: 2021-08-13 15:35:25
'''
from ctypes import cdll, c_char_p
from .PlatformHelper import PlatformHelper as ph
from wtpy.WtUtilDefs import singleton
import os

@singleton
class WtExecApi:

    # api可以作为公共变量
    api = None
    ver = "Unknown"

    def __init__(self):
        paths = os.path.split(__file__)
        dllname = ph.getModule("WtExecMon")
        a = (paths[:-1] + (dllname,))
        _path = os.path.join(*a)
        self.api = cdll.LoadLibrary(_path)

        self.api.get_version.restype = c_char_p
        self.ver = bytes.decode(self.api.get_version())

    def run(self):
        self.api.run_exec()

    def release(self):
        self.api.release_exec()

    def write_log(self, level:int, message:str, catName:str = ""):
        self.api.write_log(level, bytes(message, encoding = "utf8").decode('utf-8').encode('gbk'), bytes(catName, encoding = "utf8"))

    def config(self, cfgfile:str = 'cfgexec.json', isFile:bool = True):
        self.api.config_exec(bytes(cfgfile, encoding = "utf8"), isFile)

    def initialize(self, logCfg:str = "logcfgexec.json", isFile:bool = True):
        '''
        C接口初始化
        '''
        self.api.init_exec(bytes(logCfg, encoding = "utf8"), isFile)
        self.write_log(102, "WonderTrader independent execution framework initialzied，version：%s" % (self.ver))

    def set_position(self, stdCode:str, target:float):
        self.api.set_position(bytes(stdCode, encoding = "utf8"), target)
