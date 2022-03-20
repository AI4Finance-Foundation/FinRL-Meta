from finrl_meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from test import test


def trade(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, API_KEY,
          API_SECRET, API_BASE_URL, trade_mode='backtesting',
          if_vix=True, **kwargs):
    if trade_mode == 'backtesting':
        # use test function for backtesting mode
        test(start_date, end_date, ticker_list, data_source, time_interval,
             technical_indicator_list, drl_lib, env, model_name, if_vix=True,
             **kwargs)

    elif trade_mode == 'paper_trading':
        # read parameters
        try:
            net_dim = kwargs.get("net_dimension", 2 ** 7)
            cwd = kwargs.get("cwd", "./" + str(model_name))  # current working directory
            state_dim = kwargs.get("state_dim")
            action_dim = kwargs.get("action_dim")
        except:
            raise ValueError('Fail to read parameters. Please check inputs for net_dim, cwd, state_dim, action_dim.')
        # initialize paper trading env
        AlpacaPaperTrading(ticker_list, time_interval, drl_lib, model_name,
                           cwd, net_dim, state_dim, action_dim,
                           API_KEY, API_SECRET, API_BASE_URL,
                           technical_indicator_list, turbulence_thresh=30,
                           max_stock=1e2, latency=None)
        # run paper trading
        AlpacaPaperTrading.run()

    else:
        raise ValueError("Invalid mode input! Please input either 'backtesting' or 'paper_trading'.")
