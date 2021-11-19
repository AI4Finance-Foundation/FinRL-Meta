from features import Indicator
from assessments import SimpleAssessment
from stoppers import SimpleStopper
from strategies import SimpleCTA
from envs import WtEnv, WtSubProcessEnv


class SimpleCTAEnv(WtEnv):
    def __init__(self,
                 time_range: tuple,
                 slippage: int = 0,
                 mode: int = 1
                 ):
        assets = 180000

        # 角色：数据研究人员、强化学习研究人员、策略研究人员
        # 原则：每个角色的分工模拟交易机构做隔离

        # 特征工程组件, 滚动窗口=2，根据特征工程自动生成强化学习需要的observation
        # 特征工程的因子生成绝大多数情况下（舆情因子、周期因子）不是由env负责的，所以尽量使用特征工程组件而不要在env中定义因子
        # 特征工程的因子定义和生成，主要使用者是数据研究人员
        # 特征工程的因子后处理，主要使用者是强化学习研究人员
        feature: Indicator = Indicator(
            code='DCE.c.HOT', period=Indicator.M5, roll=1, assets=assets)  # 每一个特征工程必须指定一个主要标的

        # 按需添加其他标的
        feature.addSecurity(code='DCE.cs.HOT')
        feature.addSecurity(code='DCE.m.HOT')
        # feature.addSecurity(code='CZCE.RM.HOT')
        # feature.addSecurity(code='CZCE.JR.HOT')
        # feature.addSecurity(code='CZCE.TA.HOT')
        # feature.addSecurity(code='DCE.jd.HOT')
        # feature.addSecurity(code='SHFE.rb.HOT')
        # feature.addSecurity(code='SHFE.hc.HOT')
        # feature.addSecurity(code='SHFE.bu.HOT')
        # feature.addSecurity(code='SHFE.fu.HOT')
        # feature.addSecurity(code='SHFE.ni.HOT')

        # 分别使用5分钟、15分钟、日线建立多周期因子
        for period in (feature.M5, ):  # feature.M5, feature.M10,
            # feature.price(period)
            # feature.volume(period)
            # feature.roc(period)
            feature.bollinger(period)  # 标准差通道
            # feature.sar(period)
            # feature.trange(period)  # 波动率
            feature.macd(period)  # 双均线强度
            # feature.rsi(period)
            # feature.dx(period)
            # feature.obv(period)
            feature.kdj(period)

        # 除上述特征，特征工程组件会自动加上 "开仓的最大浮盈、开仓的最大亏损、开仓的浮动盈亏、当前持仓数"4列，如果没有持仓则全部为0

        # 止盈止损组件，暂时是个摆设
        # 止盈止损组件的主要使用者是策略研究人员
        stopper: SimpleStopper = SimpleStopper()

        # 评估组件
        # 评估组件的主要使用者是强化学习研究人员定义reward
        assessment: SimpleAssessment = SimpleAssessment(init_assets=assets)
        super().__init__(
            # 策略只做跟交易模式相关的操作(如趋势策略、日内回转、配对交易、统计套利)，不参与特征生成和评估，主要使用者是策略研究人员
            strategy=SimpleCTA,
            stopper=stopper,
            feature=feature,  # 特征计算
            assessment=assessment,  # 评估计算
            time_range=time_range,
            slippage=slippage,
            mode=mode,  # 1训练模式，2评估模式，3debug模式
        )

    def _debug_(self):
        print('%s: assets %s, reward %s, reward_sum %s' % (
            self._name_(self._iter_), self._assessment_.curr_assets-self._assessment_.init_assets, self._assessment_.reward, sum(self._assessment_.rewards)))


class SimpleCTASubProcessEnv(WtSubProcessEnv):
    def __init__(self,
                 time_range: tuple,
                 slippage: int = 0,
                 mode: int = 1):
        super().__init__(
            cli=SimpleCTAEnv,
            time_range=time_range,
            slippage=slippage,
            mode=mode)


if __name__ == '__main__':
    env: WtEnv = SimpleCTASubProcessEnv(time_start=201901011600,
                                        time_end=202001011600, mode=2)

    print(env.action_space.contains)

    for i in range(1):  # 模拟训练10次
        obs = env.reset()
        done = False
        n = 0
        while not done:
            # box.contains of Box([-3. -3. -3. -3.], [3. 3. 3. 3.], (4,), float16)
            action = env.action_space.sample()  # 模拟智能体产生动作
            obs, reward, done, info = env.step(action)
            n += 1
            print(
                # 'action:', action,
                # 'obs:', obs,
                'reward:', reward,
                # 'done:', done
            )
        #     break
        # break
        print('第%s次训练完成，执行%s步, 市值%s。' % (i+1, n, env.assets))
    env.close()
