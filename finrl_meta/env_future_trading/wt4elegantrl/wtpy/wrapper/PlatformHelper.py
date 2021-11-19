'''
Descripttion: Automatically generated file comment
version: 
Author: Wesley
Date: 2021-02-26 16:28:57
LastEditors: Wesley
LastEditTime: 2021-08-13 15:32:23
'''
import platform

class PlatformHelper:

    @staticmethod
    def isPythonX64() -> bool:
        ret = platform.architecture()
        return (ret[0] == "64bit")

    @staticmethod
    def isWindows() -> bool:
        if "windows" in platform.system().lower():
            return True

        return False

    @staticmethod
    def getModule(moduleName:str, subdir:str="") -> str:
        dllname = ""
        ext = ""
        prefix = ""
        if PlatformHelper.isWindows(): #windows平台
            ext = ".dll"
            if PlatformHelper.isPythonX64():
                dllname = "x64/"
            else:
                dllname = "x86/"
        else:#Linux平台
            dllname = "linux/"
            prefix = "lib"
            ext = ".so"

        if subdir != "":
            dllname += subdir + "/"

        dllname += prefix + moduleName + ext
        return dllname
            