def data_fetch(start_time, end_time, stock_list, resolution=Resolution.Daily) :
    #resolution: Daily, Hour, Minute, Second
    qb = QuantBook()
    for stock in stock_list:
        qb.AddEquity(stock)
    history = qb.History(qb.Securities.Keys, start_time, end_time, resolution)
    return history