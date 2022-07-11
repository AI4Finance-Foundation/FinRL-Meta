import logging

logger = logging.getLogger()
logger.setLevel("DEBUG")
file_handler = logging.FileHandler("./log.txt", mode='a', encoding="utf-8")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(logging.Formatter(fmt="%(lineno)s---%(asctime)s---%(message)s"))
logger.addHandler(file_handler)

import jqdatasdk as jq
from neo_finrl.agents.elegantrl.agent import *
from neo_finrl.agents.elegantrl.run import *
from neo_finrl.data_processors.joinquant.data_fetch_jq import data_fetch
from neo_finrl.data_processors.joinquant.env_stock_jq import StockTradingEnv
from neo_finrl.data_processors.joinquant.preprocess_jq import preprocess

"""data_fetch"""
# please fill in your own account info
jq.auth("user_name", "password")
stock_list = [
    "000001.XSHE",
    "000002.XSHE",
    "000004.XSHE",
    "000006.XSHE",
    "000009.XSHE",
]
num = 3000
unit = "15m"
end_dt = "2021-05-10"
jq_df = data_fetch(stock_list, num, unit, end_dt)
jq_ary = preprocess(jq_df, stock_list)
logging.info(jq_ary.shape)
args = Arguments(if_on_policy=True, gpu_id=0)
args.agent = AgentPPO()

"""choose environment"""
args.env = StockTradingEnv(ary=jq_ary, if_train=True)
args.env_eval = StockTradingEnv(ary=jq_ary, if_train=False)
args.net_dim = 2**9  # change a default hyper-parameters
args.batch_size = 2**8
args.break_step = int(5e5)

train_and_evaluate(args)
