import numpy as np
import numpy.random as rd
import pandas as pd
import torch as th

Ary = np.ndarray
DataDir = "./dlmd"

"""load csv"""


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
    data = type("", (), {})()
    time_col = "ExchangeTS"

    data.ins_id = df.iloc[0]["InstrumentID"]
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


"""week factors"""


def moving_average(ary0, k):
    window = np.ones(k) / k

    ary1 = ary0.copy()
    ary1[k - 1 :] = np.convolve(ary0, window, mode="valid")
    for i in range(k):
        ary1[i] = ary0[: i + 1].mean()
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

    press1 = ap1 * av1 - bp1 * bv1
    press1_px = press1 / (av1 + bv1 + 1)
    press1_pv = press1 / wol
    delta1_px = press1_px - px0

    amount0 = px0 * wol
    amount1 = press1_px * (rate_av1 + rate_bv1)
    weight_px = (amount1 + amount0) / wol
    weight_vpx = (amount1 + amount0) / (rate_av1 + rate_bv1 + wol)

    week_factors = []
    for ary in (
        px0,
        vol,
        rate_av1,
        rate_bv1,
        press1,
        press1_px,
        press1_pv,
        delta1_px,
        amount0,
        amount1,
        weight_px,
        weight_vpx,
    ):
        week_factors.extend(get_arys_move_average_05_15_30(ary))
    return np.stack(week_factors, axis=1)


"""env"""


class FutureTradingVecEnv:
    def __init__(
        self, num_envs=8, cost_pct=1e-4, max_position=64, max_holding=512, gpu_id=-1
    ):
        data_dir = DataDir
        data_name = "cu_top1_2023-01-03_2023-03-07.csv"
        data_path = f"{data_dir}/{data_name}"

        self.device = th.device(
            f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )

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

        self.num_envs = num_envs
        self.cost_pct = cost_pct
        self.max_position = max_position
        self.max_holding = max_holding

        self.map_i_to_action = th.tensor((0, 1, 2, 4, -4, -2, -1), device=self.device)

        # reset()
        self.px0 = None  # close price
        self.vol = None  # volume
        self.fac = None  # factors
        self.t = None
        self.max_t = None
        self.asset = None
        self.cumulative_returns = 0

        self.amount = None
        self.position = None
        self.holding = None  # holding period

        # environment information
        self.env_name = "OptionStockEnv-v0"

        position_dim = 1
        holding_dim = 1
        factors_dim = data.week_factors.shape[1]
        self.state_dim = position_dim + holding_dim + factors_dim
        self.action_dim = len(self.map_i_to_action)
        self.if_discrete = True
        self.max_step = max([data.px0.shape[0] for data in datas])

    def reset(self):
        self.t = 0

        rd_i_data = rd.randint(len(self.datas))
        data = self.datas[rd_i_data]
        self.data = data
        self.px0 = th.tensor(data.px0, dtype=th.float32, device=self.device)
        self.vol = th.tensor(data.vol, dtype=th.float32, device=self.device)
        self.fac = th.tensor(data.week_factors, dtype=th.float32, device=self.device)
        self.max_t = data.px0.shape[0] - 1
        self.asset = th.zeros(self.num_envs, dtype=th.float32, device=self.device)

        self.amount = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        self.position = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        self.holding = th.zeros(self.num_envs, dtype=th.float32, device=self.device)

        obs = self.get_state()
        return obs, {}

    @staticmethod
    def load_datas_from_disk(data_path):
        # data_dir = DataDir
        # data_name = "cu_top1_2023-01-03_2023-03-07.csv"
        # data_path = f"{data_dir}/{data_name}"
        df_raw = pd.read_csv(data_path, parse_dates=["ExchangeTS"])
        dfs = split_df_by_time(df_raw, time_col="ExchangeTS")

        datas = [get_data_of_arys_from_df(df) for df in dfs]
        for data in datas:
            data.week_factors = get_week_factors_to_data(data)
        return datas

    def get_state(self):
        state = th.empty(
            (self.num_envs, self.state_dim), dtype=th.float32, device=self.device
        )
        state[:, 0] = self.position
        state[:, 1] = self.holding
        state[:, 2:] = self.fac[self.t]
        return state

    def step(self, action):
        self.t += 1
        close_price = self.px0[self.t]
        volume = self.vol[self.t]

        a0_int = self.map_i_to_action[action]

        # limit -max_position <= position <= +max_position
        a1_int = a0_int.clip(
            min=-self.max_position - self.position,
            max=self.max_position - self.position,
        )

        # limit the trade_action when over max_holding, write before limit trade_volume
        holding_mask = self.holding > self.max_holding
        a1_int[holding_mask] = -self.position[holding_mask]

        # limit -trade_volume <= trade_action <= +trade_volume
        trade_volume = (volume * 0.5).to(th.long)
        a1_int = a1_int.clip(min=-trade_volume, max=trade_volume)

        # limit a1_int by _position, in order to empty position
        _position = self.position + a1_int
        _position_mask = (self.position * _position) < 0
        a1_int[_position_mask] = -self.position[_position_mask]

        """update holding"""
        position = self.position + a1_int
        position_mask = position == 0
        self.holding[position_mask] = 0
        self.holding += 1

        """trade: update amount asset"""
        amount = self.amount - a1_int * close_price - th.abs(a1_int) * self.cost_pct
        asset = amount + position * close_price

        """get reward"""
        reward = asset - self.asset

        self.position = position
        self.amount = amount
        self.asset = asset

        done = self.t >= self.max_t
        if self.t >= self.max_t:
            state, info = self.reset()
        else:
            state, info = self.get_state(), {}

        terminal = position_mask  # position == 0
        truncate = th.full((4,), done, dtype=th.bool)
        return state, reward, terminal, truncate, {}


def run():
    num_envs = 16
    gpu_id = -1

    env = FutureTradingVecEnv(num_envs=num_envs, gpu_id=gpu_id)

    device = env.device
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim

    state, info = env.reset()
    assert state.shape == (num_envs, state_dim)

    cumulative_reward = th.zeros_like(env.amount)
    for t in range(max_step):
        action = th.randint(action_dim, size=(num_envs,), device=device)
        state, reward, terminal, truncate, info = env.step(action)

        cumulative_reward += reward

    print(cumulative_reward.detach().cpu().numpy().round(3))


if __name__ == "__main__":
    run()
