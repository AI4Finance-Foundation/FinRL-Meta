import numpy as np
import talib as ta
from reprocess import REPROCESS, ZFILTER, MAXMIN
from wtpy.StrategyDefs import CtaContext, HftContext


class Feature():
    M1 = 'm1'
    M3 = 'm3'
    M5 = 'm5'
    M10 = 'm10'
    M15 = 'm15'
    M30 = 'm30'
    M60 = 'm60'
    D1 = 'd1'

    def __init__(self, code: str, period: str, roll: int, assets: float = 1000000) -> None:
        self.__shape__: tuple = tuple()
        self._roll_: int = int(roll)
        self._assets_: float = float(assets)

        self.__cb__: dict = {}

        self.__obs__: dict = {}
        self.__time__: int = 0

        self.__securities__: list = []
        self.addSecurity(code=code)

        self.__main__: tuple = (code, period)
        self.__subscribies__: dict = {}
        self._subscribe_(period=period, count=1)

        # self.__comminfo__: dict = {}

    @property
    def securities(self):
        return self.__securities__

    def addSecurity(self, code: str):
        if self.__shape__ or code in self.__securities__:
            return
        self.__securities__.append(code)

    def _subscribe_(self, period: str, count: int = 1):
        self.__subscribies__[period] = max(
            self.__subscribies__.get(period, 0),
            count+self._roll_
        )

    def subscribe(self, context: CtaContext):
        '''
        根据特征需求订阅数据
        '''
        for code in self.__securities__:
            # comminfo = context.stra_get_comminfo(code)  # 品种信息数据
            # self.__comminfo__[code] = (comminfo.pricetick, comminfo.volscale)
            for period, count in self.__subscribies__.items():
                context.stra_get_bars(
                    stdCode=code,
                    period=period,
                    count=count,
                    isMain=(code == self.__main__[0]
                            and period == self.__main__[1])
                )

    def _callback_(self, space: int, period: str, callback, reprocess: REPROCESS, **kwargs):
        if self.__shape__ or space < 1:
            return
        if period not in self.__cb__:
            self.__cb__[period] = {}
        self.__cb__[period][callback.__name__] = (
            space, callback, reprocess, kwargs)

    def sigmoid(self, value, thresh=30):
        return (1 / (1 + np.exp(-(value/thresh) * np.e)) - 0.5)*thresh

    @property
    def observation(self) -> dict:
        '''
        根据特征需求生成observation
        '''
        self.__shape__ = (
            len(self.securities),
            sum(c[0] for v in self.__cb__.values()
                for c in v.values())*self._roll_+4
        )
        self.__flatten__ = (self.__shape__[0]*self.__shape__[1],)
        return dict(low=-np.inf, high=np.inf, shape=self.__flatten__, dtype=np.float64)

    def calculate(self, context: CtaContext):
        self.__time__ = context.stra_get_date()*10000+context.stra_get_time()
        if self.__time__ not in self.__obs__:
            obs = np.full(shape=self.__shape__,
                          fill_value=np.nan, dtype=np.float64)
            for i, code in enumerate(self.securities):  # 处理每一个标的
                n = 0
                for period, v in self.__cb__.items():  # 处理每一个周期
                    for space, callback, p, args in v.values():  # 处理每一个特征
                        features = callback(
                            context=context, code=code, period=period, args=args)  # 通过回调函数计算特征
                        if space == 1:
                            features = (features, )
                        for feature in features:  # 处理每一个返回值
                            # print(p.calculate(feature))
                            # obs[i][n:n +self._roll_] = p.calculate(feature)[-self._roll_:]
                            obs[i][n:n +self._roll_] = p.calculate(feature)[-self._roll_:]
                            #np.clip(p.calculate(feature)[-self._roll_:], -1, 1)
                            n += self._roll_
            # self.__obs__[self.__time__] = obs
            # self.__obs__[self.__time__] = self.sigmoid(obs)
            self.__obs__[self.__time__] = obs

        # 开仓最大浮盈
        self.__obs__[self.__time__][:, -4] = tuple(
            context.stra_get_detail_profit(
                # stdCode=code, usertag='', flag=1)/self.__comminfo__[code][1]/self.__comminfo__[code][0] for code in self.securities
                stdCode=code, usertag='', flag=1)/self._assets_ for code in self.securities
        )

        # 开仓最大亏损
        self.__obs__[self.__time__][:, -3] = tuple(
            context.stra_get_detail_profit(
                # stdCode=code, usertag='', flag=-1)/self.__comminfo__[code][1]/self.__comminfo__[code][0] for code in self.securities
                stdCode=code, usertag='', flag=-1)/self._assets_ for code in self.securities
        )

        # 开仓浮动盈亏
        self.__obs__[self.__time__][:, -2] = tuple(
            context.stra_get_detail_profit(
                # stdCode=code, usertag='', flag=0)/self.__comminfo__[code][1]/self.__comminfo__[code][0] for code in self.securities
                stdCode=code, usertag='', flag=0)/self._assets_ for code in self.securities
        )

        # 持仓数
        self.__obs__[self.__time__][:, -1] = tuple(
            context.stra_get_position(stdCode=code) for code in self.securities)

        # self.__obs__[self.__time__][:, -4:] = self.sigmoid(self.__obs__[self.__time__][:, -4:])
        # np.clip(
        #     self.__obs__[self.__time__][:, -4:], -1, 1,
        #     out=self.__obs__[self.__time__][:, -4:])

    @property
    def obs(self):
        # .astype(np.float64)
        return self.__obs__.get(self.__time__).reshape(self.__flatten__)

    def price(self, period: str, reprocess: REPROCESS = MAXMIN):
        def price(context: CtaContext, code: str, period: str, args: dict):
            return context.stra_get_bars(stdCode=code, period=period, count=self.__subscribies__[period]).closes

        self._subscribe_(period=period, count=2+reprocess.n())
        self._callback_(space=1, period=period,
                        callback=price, reprocess=reprocess)

    def volume(self, period: str, reprocess: REPROCESS = MAXMIN):
        def volume(context: CtaContext, code: str, period: str, args: dict):
            return context.stra_get_bars(stdCode=code, period=period, count=self.__subscribies__[period]).volumes

        self._subscribe_(period=period, count=2+reprocess.n())
        self._callback_(space=1, period=period,
                        callback=volume, reprocess=reprocess)


class Indicator(Feature):
    def roc(self, period: str, reprocess: REPROCESS = REPROCESS):
        def roc(context: CtaContext, code: str, period: str, args: dict):
            price = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period]).closes
            price = np.log(price)
            return (price[1:]/price[:-1]-1)
            return np.diff(context.stra_get_bars(stdCode=code, period=period, count=self.__subscribies__[period]).closes)

        self._subscribe_(period=period, count=2+reprocess.n())
        self._callback_(space=1, period=period,
                        callback=roc, reprocess=reprocess)

    def bollinger(self, period: str, timeperiod=5, nbdevup=2, nbdevdn=2, reprocess: REPROCESS = MAXMIN):
        def bollinger(context: CtaContext, code: str, period: str, args: dict):
            closes = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period]).closes
            upperband, middleband, lowerband = ta.BBANDS(closes, **args)
            return upperband, middleband, lowerband
            return upperband/closes-1, middleband/closes-1, lowerband/closes-1

        self._subscribe_(period=period, count=timeperiod+reprocess.n())
        self._callback_(space=3, period=period, callback=bollinger, reprocess=reprocess,
                        timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)

    def sar(self, period: str, acceleration=0, maximum=0, reprocess: REPROCESS = MAXMIN):
        def sar(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            return ta.SAR(high=bars.highs, low=bars.lows, **args)
            return ta.SAR(high=bars.highs, low=bars.lows, **args)/bars.closes-1
        self._subscribe_(period=period, count=10+reprocess.n())
        self._callback_(space=1, period=period, acceleration=acceleration, maximum=maximum,
                        callback=sar, reprocess=reprocess)

    def trange(self, period: str, reprocess: REPROCESS = MAXMIN):
        def trange(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            return ta.TRANGE(high=bars.highs, low=bars.lows, close=bars.closes)
            return ta.TRANGE(high=bars.highs, low=bars.lows, close=bars.closes)/bars.closes

        self._subscribe_(period=period, count=2+reprocess.n())
        self._callback_(space=1, period=period,
                        callback=trange, reprocess=reprocess)

    def macd(self, period: str, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9, reprocess: REPROCESS = MAXMIN):
        def macd(context: CtaContext, code: str, period: str, args: dict):
            return ta.MACD(np.log(context.stra_get_bars(stdCode=code, period=period, count=self.__subscribies__[period]).closes), **args)

        self._subscribe_(period=period, count=slowperiod +
                         signalperiod+reprocess.n())
        self._callback_(space=3, period=period, callback=macd, reprocess=reprocess,
                        fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    def rsi(self, period: str, fastperiod: int = 6, midperiod: int = 12, slowperiod: int = 24, reprocess: REPROCESS = MAXMIN):
        def rsi(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            return ta.RSI(bars.closes, args['fastperiod']), ta.RSI(bars.closes, args['midperiod']), ta.RSI(bars.closes, args['slowperiod'])
            return ta.RSI(bars.closes, args['fastperiod'])/100, ta.RSI(bars.closes, args['midperiod'])/100, ta.RSI(bars.closes, args['slowperiod'])/100

        self._subscribe_(period=period, count=slowperiod + 1 + reprocess.n())
        self._callback_(space=3, period=period, callback=rsi, reprocess=reprocess,
                        fastperiod=fastperiod, midperiod=midperiod, slowperiod=slowperiod)

    def dx(self, period: str, timeperiod=14, reprocess: REPROCESS = MAXMIN):
        def dx(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            return ta.DX(high=bars.highs, low=bars.lows, close=bars.closes, **args)
            return ta.DX(high=bars.highs, low=bars.lows, close=bars.closes, **args)/100

        self._subscribe_(period=period, count=timeperiod+1+reprocess.n())
        self._callback_(space=1, period=period, callback=dx, reprocess=reprocess,
                        timeperiod=timeperiod)

    def obv(self, period: str, reprocess: REPROCESS = MAXMIN):
        def obv(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            return ta.OBV(bars.closes, bars.volumes, **args)

        self._subscribe_(period=period, count=10+reprocess.n())
        self._callback_(space=1, period=period,
                        callback=obv, reprocess=reprocess)

    def kdj(self, period: str, fastk_period: int = 5, slowk_period: int = 3, reprocess: REPROCESS = MAXMIN):
        def kdj(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            k, d = ta.STOCH(high=bars.highs, low=bars.lows,
                            close=bars.closes, **args)
            return k, d, (3*k-2*d)
            return k/100, d/100, (3*k-2*d)/100

        self._subscribe_(period=period, count=10 + 1 + reprocess.n())
        self._callback_(space=3, period=period, callback=kdj, reprocess=reprocess,
                        fastk_period=fastk_period, slowk_period=slowk_period)

    # def weights(self, period: str, timeperiod:int=1, index:str='000300', reprocess:REPROCESS =REPROCESS):
    #     def example(context: CtaContext, code: str, period: str, args: dict):
    #         # 标的代码 code
    #         # 标的周期 period
    #         # 自定义参数 args['index']
    #         # 日期int context.stra_get_date()
    #         # 时间int context.stra_get_time()
    #         return 查询代码(code, context.stra_get_date(), args['index'])

    #     self._subscribe_(period=period, count=1+reprocess.n())  # 在什么周期的event触发，需要几根bar
    #     self._callback_(
    #         space=1, #查询代码有几个值，自动生成obs的占位空间
    #         period=period,
    #         callback=example,
    #         reprocess=reprocess
    #         timeperiod=timeperiod,
    #         index=index)
