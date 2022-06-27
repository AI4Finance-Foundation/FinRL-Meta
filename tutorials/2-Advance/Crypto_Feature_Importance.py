import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from IPython import display
from datetime import datetime, timedelta
import datetime as dt
import pickle

from finrl_meta import config
from finrl_meta.data_processor import DataProcessor
from main import check_and_make_directories
from finrl_meta.data_processors.binance import Binance
from talib.abstract import MACD, RSI, CCI, DX

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from IPython.display import display, HTML

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

pd.options.display.max_columns = None

print("ALL Modules have been imported!")

# %% md

### Create folders

# %%

import os

'''
use check_and_make_directories() to replace the following
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")
'''

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# %% md

### Download data, cleaning and feature engineering

# %%

ticker_list = ['BTCUSDT']
TIME_INTERVAL = '1h'
# start_date = '2015-01-01 00:00:00'
# end_date = '2020-01-01 00:00:00'

TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE= '2019-08-01'
TRADE_START_DATE = '2019-08-01'
TRADE_END_DATE = '2020-01-03'
technical_indicator_list = ['open',
                             'high',
                             'low',
                             'close',
                             'volume',
                             'macd',
                             'macd_signal',
                             'macd_hist',
                             'rsi',
                             'cci',
                             'dx'
                             ]

if_vix = False
# download and clean

p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(ticker_list=ticker_list)
p.clean_data()
df = p.dataframe
print(f"p.dataframe: {p.dataframe}")


def add_technical_indicator(df, tech_indicator_list):
    # print('Adding self-defined technical indicators is NOT supported yet.')
    # print('Use default: MACD, RSI, CCI, DX.')

    final_df = pd.DataFrame()
    for i in df.tic.unique():
        tic_df = df[df.tic == i].copy()
        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
                                                                          slowperiod=26, signalperiod=9)
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        final_df = final_df.append(tic_df)
    return final_df

processed_df=add_technical_indicator(df,technical_indicator_list)
print(f"processed_df: {processed_df.head()}")

# Drop unecessary columns and make time as index
processed_df.index=pd.to_datetime(processed_df.time)
processed_df.drop('time', inplace=True, axis=1)
print(processed_df.tail(20))


# IMPORTANT: Make sure that pd.Timedelta() is according to the time_interval to get the volatility for that time interval
# if TIME_INTERVAL == '5m':
#     Delta = pd.Timedelta(minutes=5)
# elif TIME_INTERVAL == '1h':
#     Delta = pd.Timedelta(hours=1)
# else:
#     raise ValueError('Timeframe not supported yet, please manually add!')

def get_vol(prices, span=100):
    # 1. compute returns of the form p[t]/p[t-1] - 1
    # 1.1 find the timestamps of p[t-1] values
    # price.index as datetime format
    df0 = prices.pct_change()
    # 2. estimate rolling standard deviation
    df0 = df0.ewm(span=span).std()
    return df0

data_ohlcv = processed_df.assign(volatility=get_vol(processed_df.close)).dropna()
print("data_ohlcv: ",data_ohlcv.head())

##Adding Path Dependency: Triple-Barrier Method
###The labeling schema is defined as follows:
#y = 2 : top barrier is hit first
#y = 1 : right barrier is hit first
#y = 0 : bottom barrier is hit first###
t_final=25
def get_barriers(daily_volatility=data_ohlcv['volatility'],
                 t_final=25,
                 upper_lower_multipliers=[2, 2],
                 price=data_ohlcv['close']):
    prices = price[daily_volatility.index]
    # create a container
    barriers = pd.DataFrame(columns=['days_passed',
                                     'price', 'vert_barrier', \
                                     'top_barrier', 'bottom_barrier'], \
                            index=daily_volatility.index)
    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc \
                              [daily_volatility.index[0]: day])
        # set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index) \
                and t_final != 0):
            vert_barrier = daily_volatility.index[
                days_passed + t_final]
        else:
            vert_barrier = np.nan
        # set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * \
                          upper_lower_multipliers[0] * vol
        else:
            # set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        # set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * \
                             upper_lower_multipliers[1] * vol
        else:
            # set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)

        barriers.loc[day, ['days_passed', 'price', 'vert_barrier', 'top_barrier', 'bottom_barrier']] = \
            days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier

    return barriers

barriers = get_barriers()
print("barriers: ",barriers.head())

####Function to get label for the dataset (0, 1, 2)
#0: hit the stoploss
#1: hit the time out
#2: hit the profit take
#The part in this function (commented out), allows for easy conversion to a regression analysis (currently it is classification).
# If one changes the labels to (-1, 0, 1), and change the hit on the vertical barrier to the function stated below.
# That will make hitting the profit take barrier 1, the vertical barrier a range from (-1, 1), and the stoploss barrier -1.
# This is a continuos space then.

def get_labels():

    '''
      start: first day of the window
      end:last day of the window
      price_initial: first day stock price
      price_final:last day stock price
      top_barrier: profit taking limit
      bottom_barrier:stop loss limt
      condition_pt:top_barrier touching conditon
      condition_sl:bottom_barrier touching conditon
    '''
    barriers["label_barrier"] = None
    for i in range(len(barriers.index)):
      start = barriers.index[i]
      end = barriers.vert_barrier[i]
      if pd.notna(end):
          # assign the initial and final price
          # price_initial = barriers.price[start]
          # price_final = barriers.price[end]
          # assign the top and bottom barriers
          top_barrier = barriers.top_barrier[i]
          bottom_barrier = barriers.bottom_barrier[i]

          #set the profit taking and stop loss conditons
          condition_pt = (barriers.price[start: end] >= top_barrier).any()
          condition_sl = (barriers.price[start: end] <= bottom_barrier).any()

          #assign the labels
          if condition_pt:
              barriers['label_barrier'][i] = 2
          elif condition_sl:
              barriers['label_barrier'][i] = 0
          else:
              # Change to regression analysis by switching labels (-1, 0, 1)
              # and uncommenting the alternative function for vert barrier
              barriers['label_barrier'][i] = 1
              # barriers['label_barrier'][i] = max(
              #           [(price_final - price_initial)/
              #             (top_barrier - price_initial), \
              #             (price_final - price_initial)/ \
              #             (price_initial - bottom_barrier)],\
              #             key=abs)
    return
get_labels()
print("barriers after labeling: ", barriers.head())
# Merge the barriers with the main dataset and drop the last t_final + 1 barriers (as they are too close to the end)

data_ohlcv = data_ohlcv.merge(barriers[['vert_barrier', 'top_barrier', 'bottom_barrier', 'label_barrier']], left_on='time', right_on='time')
data_ohlcv.drop(data_ohlcv.tail(t_final + 1).index,inplace = True)
print("data_ohlcv after labeling: ", data_ohlcv.head())
# Count barrier hits ( 0 = stoploss, 1 = timeout, 2 = profit take)
print(pd.Series(data_ohlcv['label_barrier']).value_counts())

###Copying the Neural Network function from AI4Finance's ActorPPO agent
#https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/net.py
data_ohlcv = data_ohlcv.drop(['vert_barrier', 'top_barrier', 'bottom_barrier','adjusted_close','tic'], axis = 1)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.state_dim = 12       # all the features
        self.mid_dim = 2**10      # net dimension
        self.action_dim = 3       # output (sell/nothing/buy)

        # make a copy of the model in ActorPPO (activation function in forward function)

        # Original initial layers
        self.fc1 = nn.Linear(self.state_dim, self.mid_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.mid_dim, self.mid_dim)

        # Original residual layers
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.mid_dim, self.mid_dim)
        self.hw1 = nn.Hardswish()
        self.fc_out = nn.Linear(self.mid_dim, self.action_dim)

    def forward(self, x):
        x = x.float()

        # Original initial layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        # Original residual layers
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.hw1(x)
        x = self.fc_out(x)
        return x

model_NN1 = Net()
print(model_NN1)


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)
# Set constants
batch_size=16
epochs=300

# Reinitiating data here
data = data_ohlcv

X = data[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'cci', 'dx', 'volatility']].values
y = np.squeeze(data[['label_barrier']].values).astype(int)

# Split into train+val and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Normalize input
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# initialize sets and convet them to Pytorch dataloader sets
train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.astype(int)).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test.astype(int)).long())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size
                          )

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1)
# Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Set optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_NN1.parameters(), lr=0.0001)


# Train function
def train(fold, model, device, trainloader, optimizer, epoch):
    model.train()
    correct_train = 0
    correct_this_batch_train = 0
    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = criterion(output, target.flatten())
        train_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\ttrain_loss: {:.6f}'.format(
                fold, epoch, batch_idx * len(data), len(train_loader.dataset),
                             100. * batch_idx / len(train_loader), train_loss.item()))

        # Measure accuracy on train set
        total_train_loss += train_loss.item()
        _, y_pred_tags_train = torch.max(output, dim=1)
        correct_this_batch_train = y_pred_tags_train.eq(target.flatten().view_as(y_pred_tags_train))
        correct_train += correct_this_batch_train.sum().item()

    return correct_train, train_loss

# Test function
def test(fold,model, device, test_loader, correct_train, train_loss):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += criterion(output, target.flatten()).item()  # sum up batch loss

          # Measure accuracy on test set
          _, y_pred_tags = torch.max(output, dim = 1)
          correct_this_batch = y_pred_tags.eq(target.flatten().view_as(y_pred_tags))
          correct += correct_this_batch.sum().item()

  test_loss /= len(test_loader.dataset)
  train_loss /= len(train_loader.dataset)

  # Print train accuracy for epoch
  # TODO: still a bug in summed up batch train loss
  print('\nTrain set for fold {}: Average train_loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
  fold, train_loss, correct_train, len(train_loader.dataset),
  100 * correct_train / len(train_loader.dataset)))

  # Print test result for epoch
  print('Test set for fold {}:  Average test_loss:  {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
      fold, test_loss, correct, len(test_loader.dataset),
      100 * correct / len(test_loader.dataset)))

model_NN1.to(device)

# State fold (no PurgedKFold build yet, ignore this)
# took about 1hour to train when epochs=300
epochs = 100
fold = 0
for epoch in range(1, epochs + 1):
  correct_train, train_loss = train(fold, model_NN1, device, train_loader, optimizer, epoch)
  test(fold, model_NN1, device, test_loader, correct_train, train_loss)


# Save model to disk and save in your own files to save you some time

#from google.colab import files

filename = 'model_NN1'
out = open(filename, 'wb')

with open(filename + '.pkl', 'wb') as fid:
  pickle.dump(model_NN1, fid)

# load pickle file
with open(filename + '.pkl', 'rb') as fid:
     model_NN1 = pickle.load(fid)

with torch.no_grad():
  # Show accuracy on test set
  model_NN1.eval()

  # predict proba
  y_pred_nn1_proba = model_NN1(torch.from_numpy(X_test).float().to(device))
  y_pred_nn1 = torch.argmax(y_pred_nn1_proba, dim=1)
  y_pred_nn1 = y_pred_nn1.cpu().detach().numpy()

# print predction values
print('labels in prediction:', np.unique(y_pred_nn1), '\n')

# print report
label_names = ['long', 'no bet', 'short']
print(classification_report(y_test.astype(int), y_pred_nn1, target_names=label_names))


def perturbation_rank(model, x, y, names):
    errors = []

    X_saved = x
    y = y.flatten()

    with torch.no_grad():
        model.eval()
        for i in range(x.shape[1]):
            # Convert to numpy, shuffle, convert back to tensor, predict
            x = x.detach().numpy()
            np.random.shuffle(x[:, i])
            x = torch.from_numpy(x).float().to(device)
            pred = model(x)

            # log_loss requires (classification target, probabilities)
            pred = pred.cpu().detach().numpy()
            error = metrics.log_loss(y, pred)
            errors.append(error)

            # Reset x to saved tensor matrix
            x = X_saved

    max_error = np.max(errors)
    importance = [e / max_error for e in errors]

    data = {'name': names, 'error': errors, 'importance': importance}
    result = pd.DataFrame(data, columns=['name', 'error', 'importance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result


names = list(data_ohlcv.columns)
names.remove('label_barrier')
rank = perturbation_rank(model_NN1,
                         torch.from_numpy(X_test).float(),
                         torch.from_numpy(y_test.astype(int)).long(),
                         names
                         )

print(display(rank))