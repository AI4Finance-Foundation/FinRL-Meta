import os
from argparse import ArgumentParser
from typing import List

from meta.config import ALPACA_API_BASE_URL
from meta.config import ALPACA_API_KEY
from meta.config import ALPACA_API_SECRET
from meta.config import DATA_SAVE_DIR
from meta.config import ERL_PARAMS
from meta.config import INDICATORS
from meta.config import RESULTS_DIR
from meta.config import RLlib_PARAMS
from meta.config import SAC_PARAMS
from meta.config import TENSORBOARD_LOG_DIR
from meta.config import TEST_END_DATE
from meta.config import TEST_START_DATE
from meta.config import TRADE_END_DATE
from meta.config import TRADE_START_DATE
from meta.config import TRAIN_END_DATE
from meta.config import TRAIN_START_DATE
from meta.config import TRAINED_MODEL_DIR
from meta.config_tickers import DOW_30_TICKER


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
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )


## Users can input the following command in terminal for different mode
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    main()
