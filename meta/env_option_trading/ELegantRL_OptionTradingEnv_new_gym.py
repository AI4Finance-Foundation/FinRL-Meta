import torch as th
import numpy as np
import numpy.random as rd
import pandas as pd

Ary = np.ndarray
DataDir = "./dlmd"

'''load csv'''


def split_df_by_time(df, time_col):
    """
    将包含日期时间列的 DataFrame 按照连续时间段分成多个小 DataFrame

    参数：
    df：包含日期时间列的 Pandas DataFrame
    time_col：日期时间列的名称

    返回值：
    一个列表，其中每个元素都是按照连续时间段分好的 DataFrame。
    """
    # 计算相邻时间值之间的差异
    diffs = df[time_col].diff()

    # 找到时间段的分界点
    breakpoints = diffs[diffs > pd.Timedelta(minutes=10)].index.tolist()

    # 按照时间段分割 DataFrame
    dfs = []
    start = 0
    for bp in breakpoints:
        dfs.append(df.iloc[start:bp])
        start = bp + 1
    dfs.append(df.iloc[start:])

    return dfs


def get_data_of_arys_from_df(df):
    data = type('', (), {})()
    time_col = 'ExchangeTS'

    data.ins_id = df.iloc[0]['InstrumentID']
    data.beg_ts = df.iloc[0][time_col]
    data.end_ts = df.iloc[-1][time_col]

    data.ap1 = df.AskPrice1.values
    data.bp1 = df.BidPrice1.values

    data.av1 = df.AskVolume1.values
    data.bv1 = df.BidVolume1.values

    data.px0 = df.LastPrice.values
    vol = df.Volume.values
    data.vol = np.insert(np.diff(vol), obj=0, values=vol[0])
    return data


'''week factors'''


def moving_average(ary0, k):
    window = np.ones(k) / k

    ary1 = ary0.copy()
    ary1[k - 1:] = np.convolve(ary0, window, mode='valid')
    for i in range(k):
        ary1[i] = ary0[:i + 1].mean()
    return ary1


def get_arys_move_average_05_15_30(ary):
    arys = []
    for i, j in ((0, 5), (5, 15), (15, 30)):
        _ary = moving_average(ary, k=j - i)
        if i == 0:
            _ary = ary
        else:
            _ary[i:] = _ary[:-i]
            _ary[:i] = 0
        arys.append(_ary)
    return arys


def get_week_factors_to_data(data):
    px0 = data.px0
    vol = data.vol

    ap1 = data.ap1
    bp1 = data.bp1

    av1 = data.av1
    bv1 = data.bv1

    wol = vol + 1
    rate_av1 = av1 / wol
    rate_bv1 = bv1 / wol

    press1 = (ap1 * av1 - bp1 * bv1)
    press1_px = press1 / (av1 + bv1 + 1)
    press1_pv = press1 / wol
    delta1_px = press1_px - px0

    amount0 = px0 * wol
    amount1 = press1_px * (rate_av1 + rate_bv1)
    weight_px = (amount1 + amount0) / wol
    weight_vpx = (amount1 + amount0) / (rate_av1 + rate_bv1 + wol)

    week_factors = []
    for ary in (
            px0, vol,
            rate_av1, rate_bv1,
            press1, press1_px, press1_pv, delta1_px,
            amount0, amount1, weight_px, weight_vpx
    ):
        week_factors.extend(get_arys_move_average_05_15_30(ary))
    return np.stack(week_factors, axis=1)


'''env'''


class OptionTradingEnv:
    def __init__(self, cost_pct=1e-4, max_position=64, gpu_id=-1):
        data_dir = DataDir
        data_name = "cu_top1_2023-01-03_2023-03-07.csv"
        data_path = f"{data_dir}/{data_name}"

        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        datas = self.load_datas_from_disk(data_path)
        self.datas = datas
        data = datas[0]
        assert isinstance(data.ins_id, str)
        assert isinstance(data.beg_ts, pd.Timestamp)
        assert isinstance(data.end_ts, pd.Timestamp)
        assert isinstance(data.px0, Ary)
        assert isinstance(data.vol, Ary)
        assert isinstance(data.ap1, Ary)
        assert isinstance(data.bp1, Ary)
        assert isinstance(data.av1, Ary)
        assert isinstance(data.bv1, Ary)
        assert isinstance(data.week_factors, Ary)
        self.data = data

        self.max_position = max_position
        self.cost_pct = cost_pct

        # reset()
        self.px0 = None
        self.vol = None
        self.fac = None
        self.t = None
        self.max_t = None
        self.total_asset = None
        self.cumulative_returns = 0

        self.amount = None
        self.position = None

        # environment information
        self.env_name = 'OptionStockEnv-v0'

        position_dim = 1
        factors_dim = data.week_factors.shape[1]
        self.state_dim = position_dim + factors_dim
        self.action_dim = 5
        self.if_discrete = True
        self.max_step = max([data.px0.shape[0] for data in datas])

    def reset(self):
        self.t = 0

        rd_i_data = rd.randint(len(self.datas))
        data = self.datas[rd_i_data]
        self.data = data
        self.px0 = data.px0
        self.vol = data.vol
        self.fac = data.week_factors
        self.max_t = data.px0.shape[0]
        self.total_asset = 0.0
        self.cumulative_returns = 0.0
        return self.get_state(), {}

    @staticmethod
    def load_datas_from_disk(data_path):
        # data_dir = DataDir
        # data_name = "cu_top1_2023-01-03_2023-03-07.csv"
        # data_path = f"{data_dir}/{data_name}"
        df_raw = pd.read_csv(data_path, parse_dates=['ExchangeTS'])

        dfs = split_df_by_time(df_raw, time_col='ExchangeTS')

        datas = [get_data_of_arys_from_df(df) for df in dfs]

        for data in datas:
            data.week_factors = get_week_factors_to_data(data)
        return datas

    def get_state(self):
        pass

    def step(self, action):
        trade_action = self.map_id_to_action(action_id=action)

        position = self.position + trade_action
        pass
        self.position = position

        total_asset = 0.0
        reward = total_asset - self.total_asset

        if self.position == 0:
            terminal = True
        else:
            terminal = False

        if self.t == self.max_t:
            truncate = True
        else:
            truncate = False
        return self.get_state(), reward, terminal, truncate, {}

    @staticmethod
    def map_id_to_action(action_id):
        # self.id_action_map = {0:-2, 1:-1, 2:0, 3:1, 4:2}
        return action_id - 2


def run():
    env = OptionTradingEnv()
    env.reset()


if __name__ == '__main__':
    run()
