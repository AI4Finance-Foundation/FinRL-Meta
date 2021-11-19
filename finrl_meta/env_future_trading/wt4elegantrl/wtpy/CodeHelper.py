import re

class CodeHelper:

    @staticmethod
    def isStdStkCode(stdCode:str) -> bool:
        pattern = re.compile("^[A-Z]+.([A-Z]+.)?\\d{6}Q?$")
        if re.match(pattern, stdCode) is not None:
            return True

        return False

    @staticmethod
    def stdCodeToStdCommID(stdCode:str) -> str:
        if CodeHelper.isStdStkCode(stdCode):
            return CodeHelper.stdStkCodeToStdCommID(stdCode)
        else:
            return CodeHelper.stdFutCodeToStdCommID(stdCode)

    @staticmethod
    def stdStkCodeToStdCommID(stdCode:str) -> str:
        ay = stdCode.split(".")
        return ay[0] + "." + "STK"

    @staticmethod
    def stdFutCodeToStdCommID(stdCode:str) -> str:
        ay = stdCode.split(".")
        return ay[0] + "." + ay[1]