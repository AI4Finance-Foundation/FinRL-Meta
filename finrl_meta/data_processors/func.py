import datetime
import os
import urllib, zipfile
from pathlib import Path
from datetime import *
from typing import List

TIME_ZONE_SHANGHAI = 'Asia/Shanghai'  ## Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = 'US/Eastern'  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = 'Europe/Paris'  # CAC,
TIME_ZONE_BERLIN = 'Europe/Berlin'  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = 'Asia/Jakarta'  # LQ45
TIME_ZONE_SELFDEFINED = 'xxx'  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)
BINANCE_BASE_URL = 'https://data.binance.vision/'

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

# e.g., '20210911' -> '2021-09-11'
def add_hyphen_for_date(d: str) -> str:
    res = d[:4] + '-' + d[4:6] + '-' + d[6:]
    return res

# e.g., '2021-09-11' -> '20210911'
def remove_hyphen_for_date(d: str) -> str:
    res = d[:4] + d[5:7] + '-' + d[8:]
    return res


# filename: str
# output: stockname
def calc_stockname_from_filename(filename):
    return filename.split("/")[-1].split(".csv")[0]


def calc_all_filenames(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    paths2 = []
    for dir in dir_list:
        filename = os.path.join(os.path.abspath(path), dir)
        if ".csv" in filename and "#" not in filename and "~" not in filename:
            paths2.append(filename)
    return paths2


def calc_stocknames(path):
    filenames = calc_all_filenames(path)
    res = []
    for filename in filenames:
        stockname = calc_stockname_from_filename(filename)
        res.append(stockname)
    return res


def remove_all_files(remove, path_of_data):
    assert remove in [0, 1]
    if remove == 1:
        os.system("rm -f " + path_of_data + "/*")
    dir_list = os.listdir(path_of_data)
    for file in dir_list:
        if "~" in file:
            os.system("rm -f " + path_of_data + "/" + file)
    dir_list = os.listdir(path_of_data)

    if remove == 1:
        if len(dir_list) == 0:
            print("dir_list: {}. Right.".format(dir_list))
        else:
            print(
                "dir_list: {}. Wrong. You should remove all files by hands.".format(
                    dir_list
                )
            )
        assert len(dir_list) == 0
    else:
        if len(dir_list) == 0:
            print("dir_list: {}. Wrong. There is not data.".format(dir_list))
        else:
            print("dir_list: {}. Right.".format(dir_list))
        assert len(dir_list) > 0


def date2str(dat):
    return datetime.date.strftime(dat, "%Y-%m-%d")


def str2date(str_dat):
    return datetime.datetime.strptime(str_dat, "%Y-%m-%d").date()

### ticker download helpers

def get_destination_dir(file_url):
    store_directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(store_directory, file_url)

def get_download_url(file_url):
    return "{}{}".format(BINANCE_BASE_URL, file_url)

#downloads zip, unzips zip and deltes zip
def download_n_unzip_file(base_path, file_name, date_range=None):
    download_path = "{}{}".format(base_path, file_name)
    if date_range:
        date_range = date_range.replace(" ","_")
        base_path = os.path.join(base_path, date_range)

    #raw_cache_dir = get_destination_dir("./cache/tick_raw")
    raw_cache_dir = "./cache/tick_raw"
    zip_save_path = os.path.join(raw_cache_dir, file_name)

    csv_name = os.path.splitext(file_name)[0]+".csv"
    csv_save_path = os.path.join(raw_cache_dir, csv_name)

    fhandles = []

    if os.path.exists(csv_save_path): 
        print("\nfile already exists! {}".format(csv_save_path))
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
            blocksize = max(4096,length//100)

        with open(zip_save_path, 'wb') as out_file:
            dl_progress = 0
            print("\nFile Download: {}".format(zip_save_path))
            while True:
                buf = dl_file.read(blocksize)   
                if not buf:
                    break
                out_file.write(buf)
                #visuals
                #dl_progress += len(buf)
                #done = int(50 * dl_progress / length)
                #sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )    
                #sys.stdout.flush()
    
        #unzip and delete zip
        file = zipfile.ZipFile(zip_save_path)
        with zipfile.ZipFile(zip_save_path) as zip:
            #guaranteed just 1 csv
            csvpath = zip.extract(zip.namelist()[0], raw_cache_dir) 
            fhandles.append(csvpath)
        os.remove(zip_save_path)
        return fhandles

    except urllib.error.HTTPError:
        print("\nFile not found: {}".format(download_url))
        pass

def convert_to_date_object(d):
    year, month, day = [int(x) for x in d.split('-')]
    date_obj = date(year, month, day)
    return date_obj

def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
    trading_type_path = 'data/spot'
    #currently just supporting spot
    if trading_type != 'spot':
        trading_type_path = f'data/futures/{trading_type}'
    if interval is not None:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
    else:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
    return path