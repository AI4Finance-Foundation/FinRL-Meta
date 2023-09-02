cfg = {
    'TRAIN_START_DATE': "2000-01-01",
    'TRAIN_END_DATE': "2020-12-31",
    'TRADE_START_DATE': "2021-01-01",
    'TRADE_END_DATE': "2023-08-31",
    'TECHNICAL_INDICATOR': [
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ],
    'save_path_data': "./dataset/tushare_qfq",
    'save_path_tic_list': "./dataset/dataset_tic_list.csv",
    "tushare_token": "89b3a4a7b139deae60a017503c40fee4bd88d8b28a5ca7b0dda1b5e8",
    'adj': "qfq"
}