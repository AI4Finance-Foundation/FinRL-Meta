from abc import abstractmethod
import numpy as np
from wtpy.StrategyDefs import CtaContext, HftContext


class Assessment():
    def __init__(self, init_assets=1000000):
        self._init_assets_ = init_assets
        self.reset()

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def calculate(self, context: CtaContext):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def done(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def curr_assets(self) -> float:
        raise NotImplementedError

    @property
    def init_assets(self) -> float:
        return self._init_assets_


class SimpleAssessment(Assessment):  # 借鉴了neofinrl
    gamma = 0.99

    def reset(self):
        self.__assets__: list = [self._init_assets_]
        self.__reward__: list = [0]
        self.__done__: bool = False
        self.__successive__: int = 1
        # self.__gamma__ = 1-self.gamma

    def calculate(self, context: CtaContext):
        if self.__done__:
            return

        # 动态权益
        dynbalance = context.stra_get_fund_data(0)
        # 总平仓盈亏
        # closeprofit = context.stra_get_fund_data(1)
        # 总浮动盈亏
        # positionprofit = context.stra_get_fund_data(2)
        # 总手续费
        # fee = context.stra_get_fund_data(3)

        self.__assets__.append(self._init_assets_+dynbalance)  # 账户实时的动态权益
        
        
        if len(self.__reward__) > 1:
            reward = (self.__assets__[-1]-self.__assets__[-2]) \
                / self._init_assets_ * 12 #* 0.382

            # if (reward < 0 and self.__reward__[-1] < 0) or \
            #         (reward > 0 and self.__reward__[-1] > 0):
            #     self.__successive__ += 1
            # else:
            #     self.__successive__ = 1
                
            # reward *= self.__successive__

            # if self.__assets__[-1] > self.__assets__[-2]: #
            #     reward += 0.0001*self.__successive__
            # else:
            #     reward -= 0.0001*self.__successive__

            # reward += (self.__assets__[-1]-max(self.__assets__[:-1])) \
            #     / self._init_assets_ * 0.1
            # reward += (self.__assets__[-1]-min(self.__assets__[:-1])) \
            #     / self._init_assets_ * 0.1

            # reward = 0

            # returns = np.diff(np.array(self.__assets__))
            # reward = (np.where(returns < 0, 0, returns).sum()-1e-5) \
            #     / abs(np.where(returns > 0, 0, returns).sum()+1e-5) \
            #     - 1
            # reward *= 0.1

            # print(np.where(returns < 0, 0, returns).sum(), np.where(returns > 0, 0, returns).sum(), dynbalance, reward)

            # 情绪奖励
            # if (reward < 0 and self.__reward__[-1] < 0) or \
            #         (reward > 0 and self.__reward__[-1] > 0):
            #     self.__successive__ += 1
            # else:
            #     self.__successive__ = 1

            # #资金成本
            # if self.__assets__[-1] > self.__assets__[-2]: #
            #     reward += 0.0001*self.__successive__
            # else:
            #     reward -= 0.0001*self.__successive__

            # reward *= 0.01

            # #近期奖励
            # if self.__assets__[-1] > max(self.__assets__[:-1]):
            #     reward += 0.02
            # if self.__assets__[-1] < min(self.__assets__[:-1]):
            #     reward -= 0.01

            # 长期奖励
            # reward += (self.__assets__[-1]-max(self.__assets__[:-1])) \
            #     / self._init_assets_ * self.__successive__ * 0.382
            # reward += (self.__assets__[-1]-min(self.__assets__[:-1])) \
            #     / self._init_assets_ * self.__successive__ * 0.382

            # if (reward < 0 and self.__reward__[-1] < 0) or \
            #         (reward > 0 and self.__reward__[-1] > 0):
            #     reward *= self.__successive__
            #     self.__successive__ += 1
            # else:
            #     self.__successive__ = 1
            # reward += (self.__assets__[-1]-max(self.__assets__[:-1])) \
            #     / self._init_assets_ * 0.382
            # reward += (self.__assets__[-1]-min(self.__assets__[:-1])) \
            #     / self._init_assets_ * 0.382

            '''
            2021/11/01

            if self.__assets__[-1] > self.__assets__[-2]:
                self.__successive__ += 1
            else:
                self.__successive__ = 1
                
            reward = -0.00001
            reward += (self.__assets__[-1] /
                    self.__assets__[-2] - 1) * 0.382 * self.__successive__
            max_assets = (self.__assets__[-1]/max(self.__assets__[:-1])-1) * 0.382
            reward += max_assets * (5 if max_assets > 0 else 1)
            min_assets = (self.__assets__[-1]/min(self.__assets__[:-1])-1) * 0.382
            reward += min_assets * (5 if min_assets < 0 else 1)
            '''
        else:
            reward = -0.01

        self.__reward__.append(reward)  # 以动态权益差分设计reward
        self.__done__ = False  # 此处可以根据控制任务结束状态

    def finish(self):
        if self.__done__:
            return

        # returns = np.add(1, self.__reward__).cumprod()
        # np.subtract(returns, 1, out=returns)

        # gamma = 0
        # for reward in np.diff(np.log(self.__assets__)):
        #     gamma *= self.gamma
        #     gamma += reward

        gamma = 0
        for reward in self.__reward__:
            gamma *= self.gamma
            gamma += reward

        # gamma = np.diff(np.array(self.__assets__))
        # gamma = (np.where(gamma < 0, 0, gamma).sum()-1e-5) \
        #     / abs(np.where(gamma > 0, 0, gamma).sum()+1e-5) \
        #     - 1

        # gamma = np.round(np.nanprod(np.array(self.__reward__)+1, axis=0)-1, 5)
        # gamma = self.__assets__[-1]/max(self.__assets__)-1
        # gamma = self.__assets__[-1]/self.init_assets-1
        self.__reward__.append(gamma)  # 在结束的时候把过程奖励做处理，作为整个训练的奖励
        self.__done__ = True

    @property
    def reward(self) -> float:
        # return self.__reward__[-1]
        return float(self.__reward__[-1])

    @property
    def rewards(self) -> float:
        return self.__reward__

    @property
    def done(self) -> float:
        return self.__done__

    @property
    def curr_assets(self) -> float:
        return self.__assets__[-1]
