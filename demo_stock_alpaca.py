from elegantrl.run import *
from neo_finrl.alpaca.preprocess_alpaca import preprocess
from neo_finrl.alpaca.data_fetch_alpaca import data_fetch

from neo_finrl.alpaca.env_stock_alpaca import StockTradingEnv
from elegantrl.agent import *

'''data_fetch'''
#please fill in your own account info
API_KEY = None
API_SECRET = None
APCA_API_BASE_URL = None
stock_list = [
    "AAPL","MSFT","JPM"
]
start = '2021-05-01'
end = '2021-05-10'
time_interval = '1Min'
alpaca_df = data_fetch(API_KEY, API_SECRET, APCA_API_BASE_URL,stock_list, start,
               end,time_interval)
alpaca_ary = preprocess(alpaca_df, stock_list)
print(alpaca_ary.shape)
args = Arguments(if_on_policy=True, gpu_id=0)
args.agent = AgentPPO()

'''choose environment'''
args.env = StockTradingEnv(ary = alpaca_ary, if_train=True)
args.env_eval = StockTradingEnv(ary = alpaca_ary, if_train=False)
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
args.break_step = int(5e5)

train_and_evaluate(args)
