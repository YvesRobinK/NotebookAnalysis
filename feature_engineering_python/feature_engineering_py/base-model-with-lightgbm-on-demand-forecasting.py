#!/usr/bin/env python
# coding: utf-8

# ## Store Item Demand Forecasting Challenge :
# 
# Dataset of the "Store Item Demand Forecasting Challenge" (https://www.kaggle.com/c/demand-forecasting-kernels-only/) is a time series related case study for me during my "Data Science and Machine Learning" bootcamp journey.
# 
# I develop a LigthGBM model including advanced feature engineering about different type approaches ie. smoothings, lag/shift injections on series, linear/nonlinear forecasting projections, encodings, model tuning etc. My notebook is able to reach public scores between 13.84000 - 13.87000. I would like to share it and any discussion/comment is welcome.
# 
# I use couple of shared notebooks to improve my notebook, and a list of them here:
# * https://www.kaggle.com/ashishpatel26/keeping-it-simple-by-xyzt
# * https://www.kaggle.com/miladdoostan/handling-outliers-feature-engineering-lgbm
# * https://www.kaggle.com/ymatioun/simple-lightgbm
# * https://www.kaggle.com/elitcohen/store-sales-eda-and-linear-drift-prediction
# * https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model
# * https://www.kaggle.com/ekrembayar/store-item-demand-forecasting-with-lgbm
# 

# In[1]:


# !pip install lightgbm==2.3.1
# import lightgbm
# lightgbm.__version__


# ### Importing Libraries and Loading Datasets :

# In[2]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import time
start_time = time.time()
warnings.filterwarnings('ignore')

# load datasets
train = pd.read_csv(r'../input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'], index_col=['date'])
test = pd.read_csv(r'../input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'], index_col=['date'])


# ### Non-Linear Growth Rate Projection for 2018 :
# #### *(data pre-processing)*
# 
# The next cell is adapted from a notebook of https://www.kaggle.com/ashishpatel26/keeping-it-simple-by-xyzt. Basically, with a nonlinear growth rate projection the sales prediction for 2018 is calculated with a high precision on reference tables of monthly, store and day of week sales by considering a full year. Quite smart solution! I attempt similar solutions mostly focus on first three months of year, but I could not achieve better than ~14.1 even though I fully calibrate day of week between each year. I use this sale prediction to expand volume of train dataset with also test dataset in the LightGBM model. 

# In[3]:


def sales_prediction():

    # Expand dataframe with more useful columns
    def expand_df(dataframe):
        dataframe['day'] = dataframe.index.day
        dataframe['month'] = dataframe.index.month
        dataframe['year'] = dataframe.index.year
        dataframe['dayofweek'] = dataframe.index.dayofweek
        return dataframe

    data = expand_df(train)

    # Only data 2015 and after is used
    new_data = data.loc[data.year >= 2015]
    grand_avg = new_data.sales.mean()

    # Day of week - Item Look up table
    dow_item_table = pd.pivot_table(new_data, index='dayofweek', columns='item', values='sales', aggfunc=np.mean)

    # Month pattern
    month_table = pd.pivot_table(new_data, index='month', values='sales', aggfunc=np.mean) / grand_avg

    # Store pattern
    store_table = pd.pivot_table(new_data, index='store', values='sales', aggfunc=np.mean) / grand_avg

    # weighted growth rate
    year_table = pd.pivot_table(data, index='year', values='sales', aggfunc=np.mean) / grand_avg
    years = np.arange(2013, 2019)
    annual_growth = np.poly1d(np.polyfit(years[:-1], year_table.values.squeeze(), 2, w=np.exp((years - 2018) / 10)[:-1]))

    pred_sales = []
    for _, row in test.iterrows():
        dow, month, year = row.name.dayofweek, row.name.month, row.name.year
        item, store = row['item'], row['store']
        base_sales = dow_item_table.at[dow, item]
        mul = month_table.at[month, 'sales'] * store_table.at[store, 'sales']
        pred_sales.append(int(np.round(base_sales * mul * annual_growth(year), 0)))

    return pred_sales


# extending train dataset with test dataset by sale prediction for 2018
test['sales'] = sales_prediction()
train = train.loc[train.index.year >= 2015, :] # use only data after 2015
df = pd.concat([train, test], sort=False)
df.reset_index(inplace=True)


# ### Generating Datetime Related Features :

# In[4]:


# create feature from datetime columns
def create_date_features(dataframe):
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day_of_month'] = dataframe.date.dt.day
    dataframe['day_of_year'] = dataframe.date.dt.dayofyear
    dataframe['week_of_year'] = dataframe.date.dt.weekofyear
    dataframe['day_of_week'] = dataframe.date.dt.dayofweek + 1
    dataframe['year'] = dataframe.date.dt.year
    dataframe['is_wknd'] = dataframe.date.dt.weekday // 4
    dataframe['is_month_start'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.date.dt.is_month_end.astype(int)
    dataframe['quarter'] = dataframe.date.dt.quarter
    dataframe['week_block_num'] = [int(x) for x in np.floor((dataframe.date - pd.to_datetime('2012-12-31')).dt.days / 7) + 1]
    dataframe['quarter_block_num'] = (dataframe['year'] - 2013) * 4 + dataframe['quarter']
    dataframe['week_of_month'] = dataframe['week_of_year'].values // 4.35
    return dataframe
                                                                                                                             
                                                                                                                                              
df = create_date_features(df)                                                                                                                 
                                                                                                                                              
# day labeling features                                                                       
df['is_Mon'] = np.where(df['day_of_week'] == 1, 1, 0)                                                                                            
df['is_Tue'] = np.where(df['day_of_week'] == 2, 1, 0)                                                                                         
df['is_Wed'] = np.where(df['day_of_week'] == 3, 1, 0)                                                                                         
df['is_Thu'] = np.where(df['day_of_week'] == 4, 1, 0)                                                                                         
df['is_Fri'] = np.where(df['day_of_week'] == 5, 1, 0)                                                                                         
df['is_Sat'] = np.where(df['day_of_week'] == 6, 1, 0)                                                                                         
df['is_Sun'] = np.where(df['day_of_week'] == 7, 1, 0)      


# ### Generating Sale Aggregation Based Feature :

# In[5]:


# generating some new features from aggregation of sales within different time frames
feat_list = ['day_of_week', 'week_of_month', 'week_of_year', 'month', 'quarter', 'is_wknd'] + ['day_of_week', 'week_of_month']
shift_values = [0, 0, 0, 0, 0, 0, 12, 12]
for time_item, shift_val in zip(feat_list, shift_values):
    grouped_df = df.groupby(['store', 'item', time_item])['sales'].expanding().mean().shift(shift_val).bfill().reset_index()
    grouped_df.columns = ['store', 'item', time_item, 'date', time_item + f'_ex_avg_sale{str(shift_val)}']
    grouped_df = grouped_df.sort_values(by=['item', 'store', 'date'])
    df[time_item + f'_ex_avg_sale{str(shift_val)}'] = grouped_df[time_item + f'_ex_avg_sale{str(shift_val)}'].values


# ### Generating Smoothing based Features with Lag/Shift, Rolling Mean and Exponentially Weighted Techniques :

# In[6]:


# make sure dataset sorted with original order                                                  
df.sort_values(by=['item', 'store', 'date'], axis=0, inplace=True) 


#generating some noise                                                                   
def random_noise(dataframe):                                                                                                                  
    return np.random.normal(scale=0.01, size=(len(dataframe),))    


# Lag/Shifted Features                                                                                                                                                      
# generating laggy features with different time windows                                                                                                                                 
def lag_features(dataframe, lags):                                                                                                            
    dataframe = dataframe.copy()                                                                                                              
    for lag in lags:                                                                                                                          
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["item", "store"])['sales'].transform(lambda x: x.shift(lag)) + random_noise(dataframe)                                                                                 
    return dataframe                                                                                                                          
                                                                                                                                              
                                                                                                                                              
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])                                                                       
                                                                                                                                 

    
# Rolling Mean Features                                                                                                                       
def roll_mean_features(dataframe, windows):                                                                                                   
    dataframe = dataframe.copy()                                                                                                              
    for window in windows:                                                                                                                    
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["item", "store"])['sales'].\
        transform(lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)            
    return dataframe                                                                                                                          
                                                                                                                                              
                                                          
df = roll_mean_features(df, [91, 182, 365, 546, 730])                                                                                         
                                                                                                                                              

    
# Exponentially Weighted Mean Features                                                                                                        
def ewm_features(dataframe, alphas, lags):                                                                                                    
    dataframe = dataframe.copy()                                                                                                              
    for alpha in alphas:                                                                                                                      
        for lag in lags:                                                                                                                      
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
            dataframe.groupby(["item", "store"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())                       
    return dataframe                                                                                                                          
                                                                                                                                              
                                                                                                                                              
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]                                             
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
df = ewm_features(df, alphas, lags)


# ### Final step for Data preparation :

# In[7]:


# One-Hot Encoding                                                                                                                            
df_dum = pd.get_dummies(df[['store', 'item', 'day_of_week', 'month', ]], columns=['store', 'item', 'day_of_week', 'month', ], dummy_na=True)  
df = pd.concat([df, df_dum], axis=1)                                                                                                          

# convert to logarithmic scale                                                                                                           
df['sales'] = np.log1p(df["sales"].values)

print(f'End of feature engineering and data preparation.') 
print(f'It takes {int(time.time()-start_time)} sec.')
print(f'---=> final dataframe has {df.shape[1]} features <=---') 


# ### LightGM Model with Final Dataset :

# In[8]:


# MODEL VALIDATION
start_time = time.time()
print("Final model calculation starts..")                                                                
cols = [col for col in df.columns if col not in ['date', 'id', "sales", "year"]]                                                           

train = df.loc[~df.sales.isna()]                                                                                                              
X_train, Y_train = train[cols], train['sales']                                                                                                                         
                                                                                                                                              
test = df.loc[df.id.notnull()]                                                                                                                
X_test = test[cols]                                                                                                                           
                                                                                                                                              
iteration = 15000
                                                                                                       
lgb_params = {                                                                                                                            
        'nthread': -1,
        'metric': 'mae',
        'boosting_type': 'gbdt',    
        'max_depth': 7,
        'num_leaves': 28,   
        'task': 'train',                                                                                                                      
        'objective': 'regression_l1',                                                                                                         
        'learning_rate': 0.05,                                                                                                                
        'feature_fraction': 0.9,                                                                                                              
        'bagging_fraction': 0.8,                                                                                                              
        'bagging_freq': 5,                                                                                                                    
        'lambda_l1': 0.06,                                                                                                                    
        'lambda_l2': 0.05,                                                                                                                    
        'verbose': -1,     }                                                                                                                           
                                                                                                                                              
# LightGBM dataset                                                                                                                        
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)                                                                
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=iteration)                                                              
test_preds = final_model.predict(X_test, num_iteration=iteration)
print(f'The model calculation is done in {int(time.time()-start_time)} sec.')   


# ### Generating the Submission File :

# In[9]:


# create submission file
submission = pd.DataFrame({ 'id': [*range(45000)], 'sales': np.round(np.expm1(test_preds),0) }) # turn back to normal scale
submission['sales'] = submission.sales.astype(int)
submission.to_csv('submission.csv', index=False)
print(f'OK, Submission file is created!')

