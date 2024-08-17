#!/usr/bin/env python
# coding: utf-8

# ![](https://bigdataanalyticsnews.com/wp-content/uploads/2021/04/Feature-Engineering.png)

# In[1]:


import os
import warnings
from pathlib import Path

# Basic libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pandas_profiling as pp
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.offline import init_notebook_mode, iplot
from IPython.core.display import display, HTML #To display html content in a code cell
get_ipython().run_line_magic('matplotlib', 'inline')

#This is a function that downcast the integer columns
def downcast_df_int_columns(df):
    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)
        
    if len(list_of_columns)>=1:
        max_string_length = max([len(col) for col in list_of_columns]) # finds max string length for better status printing
        print("downcasting integers for:", list_of_columns, "\n")
        
        for col in list_of_columns:
            print("reduced memory usage for:  ", col.ljust(max_string_length+2)[:max_string_length+2],
                  "from", str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8), "to", end=" ")
            df[col] = pd.to_numeric(df[col], downcast="integer")
            print(str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8))
    else:
        print("no columns to downcast")
    
    gc.collect()
    
    print("done")
    
    
#This is a function that downcast the float columns,
#if you have too many columns to adjust and do not want to see to many messages proceesing, you could comment our the print() columns
def downcast_df_float_columns(df):
    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)
        
    if len(list_of_columns)>=1:
        max_string_length = max([len(col) for col in list_of_columns]) # finds max string length for better status printing
        print("downcasting float for:", list_of_columns, "\n")
        
        for col in list_of_columns:
            print("reduced memory usage for:  ", col.ljust(max_string_length+2)[:max_string_length+2],
                  "from", str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8), "to", end=" ")
            df[col] = pd.to_numeric(df[col], downcast="float")
            print(str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8))
    else:
        print("no columns to downcast")
    
    gc.collect()
    print("done")
    

warnings.filterwarnings("ignore")


# <div class='alert alert-info'>
# <h3><center>Import the data</center></h3>
# </div>

# In[2]:


data=pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')
print('Shape of the data:',data.shape)
print('\n')
print('#of unique row_ids:',data['RowId'].nunique())
print('\n')
print('#Unique dates:',data['Date'].nunique())
print('\n')
print('# of securities:',data['SecuritiesCode'].nunique())
print('\n')


# <div class='alert alert-info'>
# <h3><center>Reduce the size of the data</center></h3>
# </div>

# In[3]:


data.info(memory_usage='deep')


# <div class='alert alert-info'>
# <h3> <center> Reducing the size of the Integer columns</center></h3>
# </div>

# In[4]:


import gc 
#Reducing Size of the Numerical dtypes

print(' \n\t\t\t\t\t\t\tReducing the size of integer columns by converting them from int64 to int32\n\n\n')
downcast_df_int_columns(data)



# <div class='alert alert-info'>
# <h3> <center> Reducing the size of the Float columns</center></h3>
# </div>

# In[5]:


print(' \n\t\t\t\t\t\t\tReducing the size of integer columns by converting them from int64 to int32\n\n\n')

downcast_df_float_columns(data)


# <div class='alert alert-info'>
# <h3> <center> The reduced size of the data</center></h3>
# </div>

# In[6]:


data.info(memory_usage='deep')

print('\n\n\n\t\t\t\t\t\tThe dataset size has been reduced from 336 MB to 251 MB')


# In[7]:


def get_month(dt):
    x = dt.strftime("%b")
    return(x)

from calendar import monthcalendar
def get_week_of_month(year, month, day):
    return next(
        (
            week_number
            for week_number, days_of_week in enumerate(monthcalendar(year, month), start=1)
            if day in days_of_week
        ),
        None,
    )

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def feature_engineering(trainer,tag):
    if tag==1: trainer.dropna(subset=['Close'],inplace=True) #Some records has nan values in closing price column
    trainer['Date']=pd.to_datetime(trainer['Date'],format='%Y-%m-%d')
    trainer['Month']=trainer['Date'].dt.month
    trainer['Year']=trainer['Date'].dt.year
    trainer['Day']=trainer['Date'].dt.day
    
    
    trainer['week_of_month']=trainer.apply(lambda x: get_week_of_month(x.Year,x.Month,x.Day),axis=1)
    trainer.drop(columns=['Month'],inplace=True)
    
    trainer['Month_name']=trainer['Date'].apply(lambda x: get_month(x))
    one_hot = pd.get_dummies(trainer['Month_name'])
    trainer=trainer.drop('Month_name',axis=1)
    trainer=trainer.join(one_hot)
    
    #trainer['Year']=trainer['Date'].dt.year.astype('category')
    #one_hot = pd.get_dummies(trainer['Year'])
    trainer=trainer.drop('Year',axis=1)
    #trainer=trainer.join(one_hot)
    
    trainer['dayofweek_num']=trainer['Date'].dt.dayofweek
    trainer['is_quater_start']=trainer['Date'].dt.is_quarter_start.map({False:0,True:1})
    trainer['is_month_start']=trainer['Date'].dt.is_month_start.map({False:0,True:1})
    trainer['is_month_end']=trainer['Date'].dt.is_month_end.map({False:0,True:1})

    #lag features
    trainer['lag_1'] = trainer['Close'].shift(1)
    trainer['lag_2'] = trainer['Close'].shift(2)
    trainer['lag_3'] = trainer['Close'].shift(3)
    trainer['lag_4'] = trainer['Close'].shift(4)
    trainer['lag_5'] = trainer['Close'].shift(5)
    trainer['lag_6'] = trainer['Close'].shift(6)
    trainer['lag_7'] = trainer['Close'].shift(7)
    
    #SMA Features
    trainer['SMA5'] = trainer.Close.rolling(5).mean()
    trainer['SMA20'] = trainer.Close.rolling(20).mean()
    trainer['SMA50'] = trainer.Close.rolling(50).mean()
    trainer['SMA200'] = trainer.Close.rolling(200).mean()
    trainer['SMA500'] = trainer.Close.rolling(500).mean()

    #EMA features
    trainer['EMA5'] = trainer.Close.ewm(span=5, adjust=False).mean()
    trainer['EMA20'] = trainer.Close.ewm(span=20, adjust=False).mean()
    trainer['EMA50'] = trainer.Close.ewm(span=50, adjust=False).mean()
    trainer['EMA200'] = trainer.Close.ewm(span=200, adjust=False).mean()
    trainer['EMA500'] = trainer.Close.ewm(span=500, adjust=False).mean()

    # Domain Specific features
    #Difference features 
    trainer['Diff_co']=trainer['Close']-trainer['Open']
    trainer['Diff_hl']=trainer['High']-trainer['Low']
    trainer['pclose']=trainer['Close'].shift(-1)
    trainer['delta']=trainer['Close']-trainer['pclose']
    trainer['daily_return']=(trainer['Close']/trainer['Open'])-1
    trainer['upper_shadow']=upper_shadow(trainer)
    trainer['lower_shadow']=lower_shadow(trainer)

    
    
    
    trainer.drop(columns=['Volume','ExpectedDividend','SupervisionFlag','AdjustmentFactor'],inplace=True)
    
    return trainer.set_index(['RowId','Date','SecuritiesCode'])


# <div class='alert alert-info'>
# <h3><center>Since the dataset has the prices of 2000 securities, I have split them 1000 securities(50%) for training, 700(35%) for validation, and 300(15%) securities for testing</center></h3>
# 
# <h6>Note: You can change the percentage of Train/Val/Test according to your strategy</h6>
# </div>

# In[8]:


unique_sec=list(data['SecuritiesCode'].unique())
print('Total number of Securities in the dataset:', len(unique_sec))
print('\n')

train=data[data['SecuritiesCode'].isin(unique_sec[0:1000])]
val=data[data['SecuritiesCode'].isin(unique_sec[1000:1700])]
test=data[data['SecuritiesCode'].isin(unique_sec[1700:len(unique_sec)])]
print('Considered training securities shape: ',train.shape)
print('Taken number of securities for training:',data[data['SecuritiesCode'].isin(unique_sec[0:1000])]['SecuritiesCode'].nunique())
print('Train data percentage(%):',round(((data[data['SecuritiesCode'].isin(unique_sec[0:1000])]['SecuritiesCode'].nunique()/len(unique_sec))*100),2))
print('\n')
print('Considered Validation securities shape: ',val.shape)
print('Considered securities in val:',val['SecuritiesCode'].nunique())
print('Validation data percentage(%):',round(((data[data['SecuritiesCode'].isin(unique_sec[1000:1700])]['SecuritiesCode'].nunique()/len(unique_sec))*100),2))
print('\n')
print('Considered testing securities shape: ',test.shape)
print('Considered securities in val:',test['SecuritiesCode'].nunique())
print('Test data percentage(%):',round(((data[data['SecuritiesCode'].isin(unique_sec[1700:len(unique_sec)])]['SecuritiesCode'].nunique()/len(unique_sec))*100),2))


# In[9]:


get_ipython().run_cell_magic('time', '', '\n#del data\n\n#IF you dont like to split the data before feature engineering\n#data=feature_engineering(data,1)\n\ntrain_copy=train.copy()\ntest_copy=test.copy()\nval_copy=val.copy()\ngc.collect()\n\ntrain=feature_engineering(train,1)\ntest=feature_engineering(test,0)\nval=feature_engineering(val,0)\n')


# <div class='alert alert-info'>
# <h3><center>Save the feature engineered dataset 📚</center></h3>
# </div>

# In[10]:


train.to_csv('train.csv')
test.to_csv('test.csv')
val.to_csv('val.csv')


# 
