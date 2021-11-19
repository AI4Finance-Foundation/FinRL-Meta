from wtpy import CtaContext, SelContext, HftContext

class BaseCtaStrategy:
    '''
    CTA策略基础类，所有的策略都从该类派生
    包含了策略的基本开发框架
    '''
    def __init__(self, name:str):
        self.__name__ = name
        
    
    def name(self) -> str:
        return self.__name__


    def on_init(self, context:CtaContext):
        '''
        策略初始化，启动的时候调用
        用于加载自定义数据

        @context    策略运行上下文
        '''
        return

    def on_session_begin(self, context:CtaContext, curTDate:int):
        '''
        交易日开始事件

        @curTDate   交易日，格式为20210220
        '''
        return

    def on_session_end(self, context:CtaContext, curTDate:int):
        '''
        交易日结束事件

        @curTDate   交易日，格式为20210220
        '''
        return
    
    def on_calculate(self, context:CtaContext):
        '''
        K线闭合时调用，一般作为策略的核心计算模块

        @context    策略运行上下文
        '''
        return

    def on_calculate_done(self, context:CtaContext):
        '''
        K线闭合时调用，一般作为策略的核心计算模块

        @context    策略运行上下文
        '''
        return


    def on_tick(self, context:CtaContext, stdCode:str, newTick:dict):
        '''
        逐笔数据进来时调用
        生产环境中，每笔行情进来就直接调用
        回测环境中，是模拟的逐笔数据

        @context    策略运行上下文
        @stdCode    合约代码
        @newTick    最新逐笔
        '''
        return

    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        '''
        K线闭合时回调

        @context    策略上下文
        @stdCode    合约代码
        @period     K线周期
        @newBar     最新闭合的K线
        '''
        return

    def on_backtest_end(self, context:CtaContext):
        '''
        回测结束时回调，只在回测框架下会触发

        @context    策略上下文
        '''
        return

class BaseHftStrategy:
    '''
    HFT策略基础类，所有的策略都从该类派生
    包含了策略的基本开发框架
    '''
    def __init__(self, name:str):
        self.__name__ = name
        
    
    def name(self) -> str:
        return self.__name__


    def on_init(self, context:HftContext):
        '''
        策略初始化，启动的时候调用
        用于加载自定义数据

        @context    策略运行上下文
        '''
        return
    
    def on_session_begin(self, context:HftContext, curTDate:int):
        '''
        交易日开始事件

        @curTDate   交易日，格式为20210220
        '''
        return

    def on_session_end(self, context:HftContext, curTDate:int):
        '''
        交易日结束事件

        @curTDate   交易日，格式为20210220
        '''
        return

    def on_backtest_end(self, context:CtaContext):
        '''
        回测结束时回调，只在回测框架下会触发

        @context    策略上下文
        '''
        return

    def on_tick(self, context:HftContext, stdCode:str, newTick:dict):
        '''
        Tick数据进来时调用

        @context    策略运行上下文
        @stdCode    合约代码
        @newTick    最新Tick
        '''
        return

    def on_order_detail(self, context:HftContext, stdCode:str, newOrdQue:dict):
        '''
        逐笔委托数据进来时调用

        @context    策略运行上下文
        @stdCode    合约代码
        @newOrdQue  最新逐笔委托
        '''
        return

    def on_order_queue(self, context:HftContext, stdCode:str, newOrdQue:dict):
        '''
        委托队列数据进来时调用

        @context    策略运行上下文
        @stdCode    合约代码
        @newOrdQue  最新委托队列
        '''
        return

    def on_transaction(self, context:HftContext, stdCode:str, newTrans:dict):
        '''
        逐笔成交数据进来时调用

        @context    策略运行上下文
        @stdCode    合约代码
        @newTrans   最新逐笔成交
        '''
        return

    def on_bar(self, context:HftContext, stdCode:str, period:str, newBar:dict):
        '''
        K线闭合时回调

        @context    策略上下文
        @stdCode    合约代码
        @period     K线周期
        @newBar     最新闭合的K线
        '''
        return

    def on_channel_ready(self, context:HftContext):
        '''
        交易通道就绪通知

        @context    策略上下文
        '''
        return

    def on_channel_lost(self, context:HftContext):
        '''
        交易通道丢失通知

        @context    策略上下文
        '''
        return

    def on_entrust(self, context:HftContext, localid:int, stdCode:str, bSucc:bool, msg:str, userTag:str):
        '''
        下单结果回报

        @context    策略上下文
        @localid    本地订单id
        @stdCode    合约代码
        @bSucc      下单结果
        @mes        下单结果描述
        '''
        return

    def on_order(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        '''
        订单回报
        @context    策略上下文
        @localid    本地订单id
        @stdCode    合约代码
        @isBuy      是否买入
        @totalQty   下单数量
        @leftQty    剩余数量
        @price      下单价格
        @isCanceled 是否已撤单
        '''
        return

    def on_trade(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, qty:float, price:float, userTag:str):
        '''
        成交回报

        @context    策略上下文
        @stdCode    合约代码
        @isBuy      是否买入
        @qty        成交数量
        @price      成交价格
        '''
        return

class BaseSelStrategy:
    '''
    选股策略基础类，所有的多因子策略都从该类派生
    包含了策略的基本开发框架
    '''
    def __init__(self, name:str):
        self.__name__ = name
        
    
    def name(self) -> str:
        return self.__name__


    def on_init(self, context:SelContext):
        '''
        策略初始化，启动的时候调用
        用于加载自定义数据

        @context    策略运行上下文
        '''
        return
    
    def on_session_begin(self, context:SelContext, curTDate:int):
        '''
        交易日开始事件

        @curTDate   交易日，格式为20210220
        '''
        return

    def on_session_end(self, context:SelContext, curTDate:int):
        '''
        交易日结束事件

        @curTDate   交易日，格式为20210220
        '''
        return
    
    def on_calculate(self, context:SelContext):
        '''
        K线闭合时调用，一般作为策略的核心计算模块
        @context    策略运行上下文
        '''
        return

    def on_calculate_done(self, context:SelContext):
        '''
        K线闭合时调用，一般作为策略的核心计算模块
        @context    策略运行上下文
        '''
        return

    def on_backtest_end(self, context:CtaContext):
        '''
        回测结束时回调，只在回测框架下会触发

        @context    策略上下文
        '''
        return

    def on_tick(self, context:SelContext, stdCode:str, newTick:dict):
        '''
        逐笔数据进来时调用
        生产环境中，每笔行情进来就直接调用
        回测环境中，是模拟的逐笔数据
        @context    策略运行上下文
        @stdCode    合约代码
        @newTick    最新逐笔
        '''
        return

    def on_bar(self, context:SelContext, stdCode:str, period:str, newBar:dict):
        '''
        K线闭合时回调
        @context    策略上下文
        @stdCode    合约代码
        @period     K线周期
        @newBar     最新闭合的K线
        '''
        return