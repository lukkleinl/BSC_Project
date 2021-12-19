import urllib.request

import pandas as pd
import yfinance as yf
import datetime as dt


class BaseLoader:
    def __init__(self, out_path):
        self.out_path = out_path

    def get_data(self):
        pass


class WineLoader(BaseLoader):
    # url for  wine quality
    # url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    def __init__(self, url, out_path):
        self.url = url
        BaseLoader.__init__(self, out_path)

    def get_data(self):
        urllib.request.urlretrieve(self.url, self.out_path)


class StockPredictionLoader(BaseLoader):
    def __init__(self, out_path):
        BaseLoader.__init__(self, out_path)

    def get_data(self, out_path):
        actual_date = dt.date.today()  # Take the actual date
        last_month_date = actual_date - dt.timedelta(days=300)
        actual_date = actual_date.strftime("%Y-%m-%d")
        last_month_date = last_month_date.strftime("%Y-%m-%d")
        # '''
        # Stock data from https://finance.yahoo.com/quote/FB/news?ltr=1
        # '''
        stock = 'FB'  # Stock name
        data = yf.download(stock, last_month_date, actual_date)  # Getting data from Yahoo Finance
        da = pd.DataFrame(data=data)
        da.to_csv(out_path)
