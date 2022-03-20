import os
import urllib
import zipfile
from datetime import *
from pathlib import Path
from typing import List

from finrl_meta.config import (
TIME_ZONE_SHANGHAI,
TIME_ZONE_USEASTERN,
TIME_ZONE_PARIS,
TIME_ZONE_BERLIN,
TIME_ZONE_JAKARTA,
TIME_ZONE_SELFDEFINED,
USE_TIME_ZONE_SELFDEFINED,
BINANCE_BASE_URL,
)

from finrl_meta.config_tickers import (
HSI_50_TICKER,
SSE_50_TICKER,
CSI_300_TICKER,
DOW_30_TICKER,
NAS_100_TICKER,
SP_500_TICKER,
LQ45_TICKER,
CAC_40_TICKER,
DAX_30_TICKER,
TECDAX_TICKER,
MDAX_50_TICKER,
SDAX_50_TICKER,
)


def calc_time_zone(ticker_list: List[str], time_zone_selfdefined: str, use_time_zone_selfdefined: int) -> str:
    time_zone = ''
    if use_time_zone_selfdefined == 1:
        time_zone = time_zone_selfdefined
    elif ticker_list in [HSI_50_TICKER, SSE_50_TICKER, CSI_300_TICKER]:
        time_zone = TIME_ZONE_SHANGHAI
    elif ticker_list in [DOW_30_TICKER, NAS_100_TICKER, SP_500_TICKER]:
        time_zone = TIME_ZONE_USEASTERN
    elif ticker_list == CAC_40_TICKER:
        time_zone = TIME_ZONE_PARIS
    elif ticker_list in [DAX_30_TICKER, TECDAX_TICKER, MDAX_50_TICKER, SDAX_50_TICKER]:
        time_zone = TIME_ZONE_BERLIN
    elif ticker_list == LQ45_TICKER:
        time_zone = TIME_ZONE_JAKARTA
    else:
        raise ValueError("Time zone is wrong.")
    return time_zone

def get_download_url(file_url):
    return f"{BINANCE_BASE_URL}{file_url}"

# downloads zip, unzips zip and deltes zip
def download_n_unzip_file(base_path, file_name, date_range=None):
    download_path = f"{base_path}{file_name}"
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)

    # raw_cache_dir = get_destination_dir("./cache/tick_raw")
    raw_cache_dir = "./cache/tick_raw"
    zip_save_path = os.path.join(raw_cache_dir, file_name)

    csv_name = os.path.splitext(file_name)[0] + ".csv"
    csv_save_path = os.path.join(raw_cache_dir, csv_name)

    fhandles = []

    if os.path.exists(csv_save_path):
        print(f"\nfile already exists! {csv_save_path}")
        return [csv_save_path]

    # make the "cache" directory (only)
    if not os.path.exists(raw_cache_dir):
        Path(raw_cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        download_url = get_download_url(download_path)
        dl_file = urllib.request.urlopen(download_url)
        length = dl_file.getheader('content-length')
        if length:
            length = int(length)
            blocksize = max(4096, length // 100)

        with open(zip_save_path, 'wb') as out_file:
            dl_progress = 0
            print(f"\nFile Download: {zip_save_path}")
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                out_file.write(buf)
                # visuals
                # dl_progress += len(buf)
                # done = int(50 * dl_progress / length)
                # sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )
                # sys.stdout.flush()

        # unzip and delete zip
        file = zipfile.ZipFile(zip_save_path)
        with zipfile.ZipFile(zip_save_path) as zip:
            # guaranteed just 1 csv
            csvpath = zip.extract(zip.namelist()[0], raw_cache_dir)
            fhandles.append(csvpath)
        os.remove(zip_save_path)
        return fhandles

    except urllib.error.HTTPError:
        print(f"\nFile not found: {download_url}")


def convert_to_date_object(d):
    year, month, day = [int(x) for x in d.split('-')]
    return date(year, month, day)


def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
    trading_type_path = 'data/spot'
    # currently just supporting spot
    if trading_type != 'spot':
        trading_type_path = f'data/futures/{trading_type}'
    return (
        f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
        if interval is not None
        else f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
    )
