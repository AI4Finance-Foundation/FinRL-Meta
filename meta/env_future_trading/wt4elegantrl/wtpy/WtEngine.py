import json

from wtpy.CtaContext import CtaContext
from wtpy.ExtModuleDefs import BaseExtExecuter
from wtpy.ExtModuleDefs import BaseExtParser
from wtpy.ExtToolDefs import BaseDataReporter
from wtpy.ExtToolDefs import BaseIndexWriter
from wtpy.HftContext import HftContext
from wtpy.SelContext import SelContext
from wtpy.StrategyDefs import BaseCtaStrategy
from wtpy.StrategyDefs import BaseHftStrategy
from wtpy.StrategyDefs import BaseSelStrategy
from wtpy.wrapper import WtWrapper
from wtpy.WtCoreDefs import EngineType
from wtpy.WtUtilDefs import singleton

from .CodeHelper import CodeHelper
from .ContractMgr import ContractInfo
from .ContractMgr import ContractMgr
from .ProductMgr import ProductInfo
from .ProductMgr import ProductMgr
from .SessionMgr import SessionInfo
from .SessionMgr import SessionMgr


@singleton
class WtEngine:
    """
    实盘交易引擎
    """

    def __init__(
        self,
        eType: EngineType,
        logCfg: str = "logcfg.json",
        genDir: str = "generated",
        bDumpCfg: bool = False,
    ):
        """
        WtEngine构造函数\n
        @eType  引擎类型：EngineType.ET_CTA、EngineType.ET_HFT、EngineType.ET_SEL\n
        @logCfg 日志配置文件\n
        @genDir 数据输出目录\n
        @bDumpCfg   是否保存最终配置文件
        """
        self.is_backtest = False

        self.__wrapper__ = WtWrapper(self)  # api接口转换器
        self.__cta_ctxs__ = dict()  # CTA策略ctx映射表
        self.__sel_ctxs__ = dict()  # SEL策略ctx映射表
        self.__hft_ctxs__ = dict()  # HFT策略ctx映射表
        self.__config__ = dict()  # 框架配置项
        self.__cfg_commited__ = False  # 配置是否已提交

        self.__writer__ = None  # 指标输出模块
        self.__reporter__ = None  # 数据提交模块

        self.__ext_parsers__ = dict()  # 外接的行情接入模块
        self.__ext_executers__ = dict()  # 外接的执行器

        self.__dump_config__ = bDumpCfg  # 是否保存最终配置

        self.__engine_type = eType
        if eType == EngineType.ET_CTA:
            self.__wrapper__.initialize_cta(logCfg=logCfg, isFile=True, genDir=genDir)
        elif eType == EngineType.ET_HFT:
            self.__wrapper__.initialize_hft(logCfg=logCfg, isFile=True, genDir=genDir)
        elif eType == EngineType.ET_SEL:
            self.__wrapper__.initialize_sel(logCfg=logCfg, isFile=True, genDir=genDir)

    def __check_config__(self):
        """
        检查设置项\n
        主要会补充一些默认设置项
        """
        if "basefiles" not in self.__config__:
            self.__config__["basefiles"] = dict()

        if "env" not in self.__config__:
            self.__config__["env"] = dict()
            self.__config__["env"]["name"] = "cta"
            self.__config__["env"]["mode"] = "product"
            self.__config__["env"]["product"] = {"session": "TRADING"}

    def getEngineType(self):
        return self.__engine_type

    def add_exetended_parser(self, parser: BaseExtParser):
        id = parser.id()
        if id not in self.__ext_parsers__:
            self.__ext_parsers__[id] = parser
            if not self.__wrapper__.create_extended_parser(id):
                self.__ext_parsers__.pop(id)

    def add_exetended_executer(self, executer: BaseExtExecuter):
        id = executer.id()
        if id not in self.__ext_executers__:
            self.__ext_executers__[id] = executer
            if not self.__wrapper__.create_extended_executer(id):
                self.__ext_executers__.pop(id)

    def get_extended_parser(self, id: str) -> BaseExtParser:
        if id not in self.__ext_parsers__:
            return None
        return self.__ext_parsers__[id]

    def get_extended_executer(self, id: str) -> BaseExtExecuter:
        if id not in self.__ext_executers__:
            return None
        return self.__ext_executers__[id]

    def push_quote_from_extended_parser(self, id: str, newTick, bNeedSlice: bool):
        self.__wrapper__.push_quote_from_exetended_parser(id, newTick, bNeedSlice)

    def set_writer(self, writer: BaseIndexWriter):
        """
        设置指标输出模块
        """
        self.__writer__ = writer

    def write_indicator(self, id: str, tag: str, time: int, data: dict):
        """
        写入指标数据
        """
        if self.__writer__ is not None:
            self.__writer__.write_indicator(id, tag, time, data)

    def set_data_reporter(self, reporter: BaseDataReporter):
        """
        设置数据报告器
        """
        self.__reporter__ = reporter

    def init(
        self,
        folder: str,
        cfgfile: str = "config.json",
        commfile: str = "commodities.json",
        contractfile: str = "contracts.json",
    ):
        """
        初始化\n
        @folder     基础数据文件目录，\\结尾\n
        @cfgfile    配置文件，json格式
        """
        f = open(cfgfile, "r")
        content = f.read()
        self.__config__ = json.loads(content)
        f.close()

        self.__check_config__()

        self.__config__["basefiles"]["commodity"] = folder + commfile
        self.__config__["basefiles"]["contract"] = folder + contractfile
        self.__config__["basefiles"]["holiday"] = folder + "holidays.json"
        self.__config__["basefiles"]["session"] = folder + "sessions.json"
        self.__config__["basefiles"]["hot"] = folder + "hots.json"

        self.productMgr = ProductMgr()
        self.productMgr.load(folder + commfile)

        self.contractMgr = ContractMgr()
        self.contractMgr.load(folder + contractfile)

        self.sessionMgr = SessionMgr()
        self.sessionMgr.load(folder + "sessions.json")

    def configEngine(self, name: str, mode: str = "product"):
        """
        设置引擎和运行模式
        """
        self.__config__["env"]["name"] = name
        self.__config__["env"]["mode"] = mode

    def addExternalCtaStrategy(self, id: str, params: dict):
        """
        添加外部的CTA策略
        """
        if "strategies" not in self.__config__:
            self.__config__["strategies"] = dict()

        if "cta" not in self.__config__["strategies"]:
            self.__config__["strategies"]["cta"] = list()

        params["id"] = id
        self.__config__["strategies"]["cta"].append(params)

    def addExternalHftStrategy(self, id: str, params: dict):
        """
        添加外部的HFT策略
        """
        if "strategies" not in self.__config__:
            self.__config__["strategies"] = dict()

        if "hft" not in self.__config__["strategies"]:
            self.__config__["strategies"]["hft"] = list()

        params["id"] = id
        self.__config__["strategies"]["hft"].append(params)

    def configStorage(self, path: str, module: str = ""):
        """
        配置数据存储\n
        @mode   存储模式，csv-表示从csv直接读取，一般回测使用，wtp-表示使用wt框架自带数据存储
        """
        self.__config__["data"]["store"]["module"] = module
        self.__config__["data"]["store"]["path"] = path

    def commitConfig(self):
        """
        提交配置\n
        只有第一次调用会生效，不可重复调用\n
        如果执行run之前没有调用，run会自动调用该方法
        """
        if self.__cfg_commited__:
            return

        cfgfile = json.dumps(self.__config__, indent=4, sort_keys=True)
        self.__wrapper__.config(cfgfile, False)
        self.__cfg_commited__ = True

        if self.__dump_config__:
            f = open("config_run.json", "w")
            f.write(cfgfile)
            f.close()

    def regCtaStraFactories(self, factFolder: str):
        """
        向底层模块注册CTA工厂模块目录\n
        !!!CTA策略只会被CTA引擎加载!!!\n
        @factFolder 工厂模块所在的目录
        """
        return self.__wrapper__.reg_cta_factories(factFolder)

    def regHftStraFactories(self, factFolder: str):
        """
        向底层模块注册HFT工厂模块目录\n
        !!!HFT策略只会被HFT引擎加载!!!\n
        @factFolder 工厂模块所在的目录
        """
        return self.__wrapper__.reg_hft_factories(factFolder)

    def regExecuterFactories(self, factFolder: str):
        """
        向底层模块注册执行器模块目录\n
        !!!执行器只在CTA引擎有效!!!\n
        @factFolder 工厂模块所在的目录
        """
        return self.__wrapper__.reg_exe_factories(factFolder)

    def addExecuter(self, id: str, trader: str, policies: dict, scale: int = 1):
        if "executers" not in self.__config__:
            self.__config__["executers"] = list()

        exeItem = {
            "active": True,
            "id": id,
            "scale": scale,
            "policy": policies,
            "trader": trader,
        }

        self.__config__["executers"].append(exeItem)

    def addTrader(self, id: str, params: dict):
        if "traders" not in self.__config__:
            self.__config__["traders"] = list()

        tItem = params
        tItem["active"] = True
        tItem["id"] = id

        self.__config__["traders"].append(tItem)

    def getSessionByCode(self, stdCode: str) -> SessionInfo:
        """
        通过合约代码获取交易时间模板\n
        @stdCode   合约代码，格式如SHFE.rb.HOT
        """
        pid = CodeHelper.stdCodeToStdCommID(stdCode)
        pInfo = self.productMgr.getProductInfo(pid)
        if pInfo is None:
            return None

        return self.sessionMgr.getSession(pInfo.session)

    def getSessionByName(self, sname: str) -> SessionInfo:
        """
        通过模板名获取交易时间模板\n
        @sname  模板名
        """
        return self.sessionMgr.getSession(sname)

    def getProductInfo(self, stdCode: str) -> ProductInfo:
        """
        获取品种信息\n
        @stdCode   合约代码，格式如SHFE.rb.HOT
        """
        return self.productMgr.getProductInfo(stdCode)

    def getContractInfo(self, stdCode: str) -> ContractInfo:
        """
        获取品种信息\n
        @stdCode   合约代码，格式如SHFE.rb.HOT
        """
        return self.contractMgr.getContractInfo(stdCode)

    def getAllCodes(self) -> list:
        """
        获取全部合约代码
        """
        return self.contractMgr.getTotalCodes()

    def add_cta_strategy(self, strategy: BaseCtaStrategy):
        """
        添加CTA策略\n
        @strategy   策略对象
        """
        id = self.__wrapper__.create_cta_context(strategy.name())
        self.__cta_ctxs__[id] = CtaContext(id, strategy, self.__wrapper__, self)

    def add_hft_strategy(
        self, strategy: BaseHftStrategy, trader: str, agent: bool = True
    ):
        """
        添加HFT策略\n
        @strategy   策略对象
        """
        id = self.__wrapper__.create_hft_context(strategy.name(), trader, agent)
        self.__hft_ctxs__[id] = HftContext(id, strategy, self.__wrapper__, self)

    def add_sel_strategy(
        self, strategy: BaseSelStrategy, date: int, time: int, period: str
    ):
        id = self.__wrapper__.create_sel_context(strategy.name(), date, time, period)
        self.__sel_ctxs__[id] = SelContext(id, strategy, self.__wrapper__, self)

    def get_context(self, id: int):
        """
        根据ID获取策略上下文\n
        @id     上下文id，一般添加策略的时候会自动生成一个唯一的上下文id
        """
        if self.__engine_type == EngineType.ET_CTA:
            if id not in self.__cta_ctxs__:
                return None

            return self.__cta_ctxs__[id]
        elif self.__engine_type == EngineType.ET_HFT:
            if id not in self.__hft_ctxs__:
                return None

            return self.__hft_ctxs__[id]
        elif self.__engine_type == EngineType.ET_SEL:
            if id not in self.__sel_ctxs__:
                return None

            return self.__sel_ctxs__[id]

    def run(self):
        """
        运行框架
        """
        if not self.__cfg_commited__:  # 如果配置没有提交，则自动提交一下
            self.commitConfig()

        self.__wrapper__.run()

    def release(self):
        """
        释放框架
        """
        self.__wrapper__.release()

    def on_init(self):
        if self.__reporter__ is not None:
            self.__reporter__.report_init_data()
        return

    def on_schedule(self, date: int, time: int, taskid: int = 0):
        # print("engine scheduled")
        if self.__reporter__ is not None:
            self.__reporter__.report_rt_data()

    def on_session_begin(self, date: int):
        # print("session begin")
        return

    def on_session_end(self, date: int):
        if self.__reporter__ is not None:
            self.__reporter__.report_settle_data()
        return
