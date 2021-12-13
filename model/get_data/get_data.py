import pandas as pd                       # structures and data analysis
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def get_data():
    #url for  wine quality
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    #get and save data to csv
    out_path='model/data/raw_data.csv'
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

def data_transformation() :
    # x = df[['High', 'Low', 'Open', 'Volume']].values  # x features
    # y = df['Close'].values                            # y labels
    wine = pd.read_csv('model/data/raw_data.csv')
    bins = (2, 6.5, 8)
    group_names = ["bad", "good"]
    wine["quality"] = pd.cut(wine["quality"], bins=bins, labels=group_names)
    label_quality = LabelEncoder()
    wine["quality"] = label_quality.fit_transform(wine["quality"])

def train_test_split(wine,test_size,random_state) :
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28) # Segment the data
    # ss = StandardScaler()                                 # Standardize the data set
    # x_train = ss.fit_transform(x_train)
    # x_test = ss.transform(x_test)
    # x_train[0:100]
    train, test = train_test_split(wine, test_size=test_size, random_state=random_state)
    train.to_csv("model/data/train.csv", index=False)
    test.to_csv("model/data/test.csv", index=False)
    

if __name__=='__main__':
    get_data()
