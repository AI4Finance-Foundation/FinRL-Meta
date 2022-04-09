import os
from typing import List
from argparse import ArgumentParser
from finrl_meta import config
from finrl_meta.config_tickers import DOW_30_TICKER
from finrl_meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)



def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


# "./" will be added in front of each directory
def check_and_make_directories(directories: List[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)



def main():
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])




## Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    main()
