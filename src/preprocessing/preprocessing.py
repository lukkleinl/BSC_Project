import urllib.request

import numpy as np
import pandas as pd


def get_data(url, out_path):
    # url for  wine quality
    # url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    # get and save data to csv
    # out_path = "/home/lukas/PycharmProjects/BSC_Project/src/data/raw_data.csv"
    urllib.request.urlretrieve(url, out_path)

    # actual_date = dt.date.today()                            # Take the actual date
    # last_month_date = actual_date-dt.timedelta(days=300) 
    # actual_date = actual_date.strftime("%Y-%m-%d") 
    # last_month_date = last_month_date.strftime("%Y-%m-%d")
    # '''
    # Stock data from https://finance.yahoo.com/quote/FB/news?ltr=1
    # '''
    # stock='FB'                                               # Stock name
    # data = yf.download(stock, last_month_date, actual_date)  # Getting data from Yahoo Finance
    # da= pd.DataFrame(data=data)
    # da.to_csv('model/data/raw_data.csv')


def replace_spaces_with_underscores(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('"', '')
    df.to_csv(path,index=False)



def data_quick_check(df):
    print(df.info())
    print(df.describe())
    print(df.head())

