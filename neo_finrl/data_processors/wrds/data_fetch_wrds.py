import wrds
import datetime
import pandas as pd

def wrds_fetch_data(dates = ['2021-05-03','2021-05-04','2021-05-05',
                             '2021-05-06','2021-05-07','2021-05-10'], 
                    stock_set =('AAPL','SPY','IBM'), time_interval = 60):
    db = wrds.Connection()
    
    def data_fetch_wrds(date='2021-05-01',
                        stock_set=('AAPL'), time_interval=60):
        #start_date, end_date should be in the same year
        current_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        lib = 'taqm_' + str(current_date.year) #taqm_2021
        table = 'ctm_'+ current_date.strftime('%Y%m%d') #ctm_20210501
        dataset = pd.DataFrame(columns = ['date','time_m','sym_root','price'])
        time_i = []
        parm = {'syms' : stock_set, 'num_shares': 300}
        data = db.raw_sql("select * from " + lib + '.'+ table + 
                          " where sym_root in %(syms)s and time_m between '9:29:01' and '16:00:59' and size > %(num_shares)s",
                          params = parm)
        data = data[['date','time_m','sym_root','price']]
        num_row = data.shape[0]
        current_time = datetime.datetime.strptime('09:30:00', '%H:%M:%S')
        for i in range(0, num_row):
            row = data.iloc[i]
            if i >= 1:
                if row['sym_root'] != data.iloc[i-1]['sym_root']:
                    current_time = datetime.datetime.strptime('09:30:00', '%H:%M:%S')
            if len(str(row['time_m'])) != 8:
                time = datetime.datetime.strptime(str(row['time_m']), '%H:%M:%S.%f')
            else:
                time = datetime.datetime.strptime(str(row['time_m']), '%H:%M:%S')
            if time > current_time:
                if row['sym_root'] != data.iloc[i-1]['sym_root']:
                    dataset = dataset.append(data.iloc[i], ignore_index = True)
                else:
                    dataset = dataset.append(data.iloc[i-1], ignore_index = True)
                time_i.append(str(current_time)[-8:])
                current_time = current_time + datetime.timedelta(seconds=time_interval)
        return dataset,time_i
    
    sets = []
    for i in dates:
        x = data_fetch_wrds(i, stock_set, 
                              time_interval)
        dataset = x[0]
        time = x[1]
        dataset['time_i'] = time
        sets.append(dataset)
        print('Data for date: ' + i + ' finished')
    result = pd.concat(sets)
    result = result.sort_values(by=['date','time_i','sym_root'])
    result = result.reset_index(drop=True)
    return result


