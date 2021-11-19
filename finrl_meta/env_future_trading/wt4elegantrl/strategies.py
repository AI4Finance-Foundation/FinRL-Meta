from features import Feature
from stoppers import Stopper
from abc import abstractmethod
from gym.spaces import Space, Box, Discrete, MultiDiscrete
from assessments import Assessment
from wtpy.WtBtEngine import EngineType
from wtpy.StrategyDefs import BaseCtaStrategy, CtaContext, BaseHftStrategy, HftContext
from numpy import around, float32


class StateTransfer():
    @staticmethod
    @abstractmethod
    def Name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def EngineType() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def Action(size: int) -> dict:
        raise NotImplementedError

    @staticmethod
    def setAction(self, action):
        raise NotImplementedError

    def __init__(self, feature: Feature, assessment: Assessment, stopper: Stopper):
        self._feature_: Feature = feature
        self._assessment_: Assessment = assessment
        self._stopper_: Stopper = stopper

        # print('StateTransfer')


class SimpleCTA(BaseCtaStrategy, StateTransfer):
    @staticmethod
    def Name() -> str:
        return __class__.__name__

    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_CTA

    @staticmethod
    def Action(size: int) -> Space:
        # return Discrete(10)
        return Box(low=-1., high=1., shape=(size, ), dtype=float32)
        # return MultiDiscrete([11]*size)
        # return dict(low=-1., high=1., shape=(size, ), dtype=float32)

    def setAction(self, action):
        # print('setAction 1')
        # action -= 5
        # self._action_ = dict(zip(self._feature_.securities, [action-5]))
        self._action_ = dict(zip(self._feature_.securities, around(action*3, 0)))
        # print(self._action_)
        # try:
        #     self._action_ = dict(zip(self._feature_.securities, around(action, 0)))
        #     print(self.name(), action, type(action))
        # except:
        #     print(self.name(), action, type(action))
        # print('setAction 2')

    def __init__(self, name: str, feature: Feature, assessment: Assessment, stopper: Stopper):
        super(BaseCtaStrategy, self).__init__(
            feature=feature, assessment=assessment, stopper=stopper)
        super().__init__(name)
        self._action_: dict = {}
        # print('TrainCTA')

    def on_init(self, context: CtaContext):
        # print('on_init 1')
        self._feature_.subscribe(context)
        # print('on_init 2')

    def on_session_begin(self, context: CtaContext, curTDate: int):
        # print('on_session_begin')
        pass

    def on_backtest_end(self, context: CtaContext):
        # print('on_backtest_end')
        pass

    def on_calculate(self, context: CtaContext):
        # print('on_calculate 1')
        self._feature_.calculate(context=context)
        self._assessment_.calculate(context=context)
        # print('on_calculate 2')

    def on_calculate_done(self, context: CtaContext):
        # print('on_calculate_done 1')
        for code in tuple(self._action_.keys()):
            qty = self._action_.pop(code)
            if qty != context.stra_get_position(stdCode=code):
                context.stra_set_position(stdCode=code, qty=qty)
            # print('stra_set_position %s'%code)
        # print('on_calculate_done 2')


# class SimpleHFT(BaseHftStrategy, StateTransfer):
#     @staticmethod
#     def Name() -> str:
#         return __class__.__name__

#     @staticmethod
#     def EngineType() -> int:
#         return EngineType.ET_HFT

#     def on_tick(self, context: HftContext, stdCode: str, newTick: dict):
#         pass
