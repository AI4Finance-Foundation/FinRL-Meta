import requests
import json
import pandas as pd
import datetime
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)


data['Time Series (Daily)']['2022-09-30']
'1. open'
'2. high'
'3. low'
'4. close'
'5. volume'

data['Meta Data']
'2. Symbol'
'5. Time Zone'
data1 = json.dumps(data['Time Series (Daily)'])
# gnData = json.dumps(data["Data"]["gn"])
df1 = pd.read_json(data1)
# gnDf = pd.read_json(gnData)

df2 = pd.DataFrame(df1.values.T, columns=df1.index, index=df1.columns)
df2.rename(columns={'1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'},
            inplace=True)
df2['tic'] = 'IBM'
# hyDf = hyDf.set_index("IndexCode")
# gnDf = gnDf.set_index("IndexCode")
dates = list(df2.index)
# 合并数据
print(df2)

