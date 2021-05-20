from elegantrl.run import *
from neo_finrl.wrds.preprocess_wrds import preprocess
from neo_finrl.wrds.data_fetch_wrds import wrds_fetch_data

from neo_finrl.wrds.env_stock_wrds import StockTradingEnv
from elegantrl.agent import *

'''data_fetch'''
dates = ['2021-05-03','2021-05-04','2021-05-05',
         '2021-05-06','2021-05-07','2021-05-10']
stock_set = ('AAPL','SPY','IBM')
df = wrds_fetch_data(dates,stock_set, time_interval=60)
a = preprocess(df)


args = Arguments(if_on_policy=True, gpu_id=0)
args.agent = AgentPPO()

'''choose environment'''
args.env = StockTradingEnv(ary = a, if_train=True)
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
args.break_step = int(5e5)

train_and_evaluate(args)
