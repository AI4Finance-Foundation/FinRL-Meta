import json

class ContractInfo:

    def __init__(self):
        self.exchg = ''     #交易所
        self.code = ''      #合约代码
        self.name = ''      #合约名称
        self.product = ''   #品种代码
        self.stdCode = ''   #标准代码

class ContractMgr:

    def __init__(self):
        self.__contracts__ = dict()

    def load(self, fname:str):
        '''
        从文件加载品种信息
        '''
        f = open(fname, 'r', encoding="gbk")
        content = f.read()
        f.close()

        exchgMap = json.loads(content)
        for exchg in exchgMap:
            exchgObj = exchgMap[exchg]

            for code in exchgObj:
                cObj = exchgObj[code]
                cInfo = ContractInfo()
                cInfo.exchg = exchg
                cInfo.code = code
                cInfo.name = cObj["name"]
                cInfo.product = cObj["product"]
                
                #股票标准代码为SSE.000001，期货标准代码为SHFE.rb.2010
                if cInfo.code[:len(cInfo.product)] == cInfo.product:
                    cInfo.stdCode = exchg + "." + cInfo.product + "." + cInfo.code[len(cInfo.product):]
                else:
                    cInfo.stdCode = exchg + "." + cInfo.code

                key = "%s.%s" % (exchg, code)
                self.__contracts__[key] = cInfo

    def getContractInfo(self, stdCode:str) -> ContractInfo:
        if stdCode[-1] == 'Q':
            stdCode = stdCode[:-1]
        else:
            items = stdCode.split(".")
            if len(items) == 3:
                stdCode = items[0] + "." + items[1] + items[2]
        if stdCode not in self.__contracts__:
            return None
            
        return self.__contracts__[stdCode]

    def getTotalCodes(self) -> list:
        codes = list()
        for code in self.__contracts__:
            codes.append(self.__contracts__[code].stdCode)
        return codes
        

