#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import time
from itertools import product


# # Loading Data

# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


INPUTFOLDER = '../input/competitive-data-science-predict-future-sales/'

item_categories = pd.read_csv(os.path.join(INPUTFOLDER, 'item_categories.csv'))
items           = pd.read_csv(os.path.join(INPUTFOLDER, 'items.csv'))
sales           = pd.read_csv(os.path.join(INPUTFOLDER, 'sales_train.csv'))
shops           = pd.read_csv(os.path.join(INPUTFOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(INPUTFOLDER, 'test.csv'))


# # Exploratory Data Analysis
# * Plot the number of items sold from Jan to Dec for the years 2013, 2014 and 2015:

# In[4]:


MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
LINEWIDTH=2
ALPHA=.6

dfp = sales[['date', 'date_block_num','item_cnt_day']].copy()

# Extract the year and the month from the date column into indepedent columns
dfp['date']  = pd.to_datetime(dfp['date'], format='%d.%m.%Y')
dfp['year']  = dfp['date'].dt.year
dfp['month'] = dfp['date'].dt.month
dfp.drop(['date'], axis=1, inplace=True)

# Sum the number of sold items for each date_block_num (which is the consecutive month number from January 2013 to October 2015)
dfp = dfp.groupby('date_block_num', as_index=False)\
       .agg({'year':'first', 'month':'first', 'item_cnt_day':'sum'})\
       .rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=False)

plt.figure(figsize=(16,6))
# Plot the sales of the year 2013
plt.plot(MONTHS, dfp[dfp.year==2013].item_cnt_month, '-o', color='steelblue', linewidth=LINEWIDTH, alpha=ALPHA,label='2013')

# Plot the sales of the year 2014
plt.plot(MONTHS, dfp[dfp.year==2014].item_cnt_month, '-o', color='seagreen', linewidth=LINEWIDTH, alpha=ALPHA,label='2014')

# Plot the sales of the year 2015 until October
plt.plot(MONTHS[:10], dfp[dfp.year==2015].item_cnt_month, '-o', color='maroon', linewidth=LINEWIDTH, alpha=ALPHA,label='2015')

# Capturing the trend between October and November (For year 2013 and 2014)
delta_2013 = dfp.iloc[10].item_cnt_month - dfp.iloc[9].item_cnt_month
delta_2014 = dfp.iloc[22].item_cnt_month - dfp.iloc[21].item_cnt_month
avg_delta = (delta_2013 + delta_2014) / 2
# Add the average to the previous month (October 2015)
nov_2015 = dfp.iloc[33].item_cnt_month + avg_delta

# MONTHS[9:11] equals ['Oct', 'Nov']
plt.plot(MONTHS[9:11], [dfp.iloc[33].item_cnt_month, nov_2015], '--o', color='gray', linewidth=LINEWIDTH, alpha=ALPHA, label='Prediction', zorder=-1)

# Axes parameters
ax = plt.gca()
ax.set_title('Sales per month')
ax.set_ylabel('# of items')
ax.grid(axis='y', color='gray', alpha=.2)
    
# Remove the frame off the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.legend(loc=2, title='Legend')
plt.show()

del dfp


# * Best Selling Categories for each year

# In[5]:


# Top N 
N=15

def get_ratio(year, topn, N):
    # Get total sold items for each year
    total = dfp.loc[year].item_cnt_year.sum()
    ratio = topn/total*100
    return "{0}: the total of the top {1} best selling items is {2} over a total of {3} for that year, which represents {4:.2f}%".format(year, N, topn, total, ratio)

dfp = sales[['date', 'item_id', 'item_cnt_day']].copy()
cats = item_categories.copy()

# Extract the year from the date column
dfp['year'] = pd.to_datetime(dfp['date'], format='%d.%m.%Y').dt.year
dfp.drop('date', axis=1, inplace=True)
dfp.item_cnt_day = dfp.item_cnt_day.astype(int)

# Remove returns
dfp = dfp[dfp.item_cnt_day>0]

# Add the category of each item
dfp = dfp.merge(items[['item_id','item_category_id']], how='left', on='item_id')

# Number of categories sold each year
dfp = dfp.groupby(['year', 'item_category_id'])\
       .agg({'item_cnt_day':'sum'})\
       .rename(columns={'item_cnt_day':'item_cnt_year'}, inplace=False)

# Top N categories sold 
top = dfp['item_cnt_year'].groupby('year', group_keys=False).nlargest(N)
# Convert top to a dataframe
top = pd.DataFrame(top).reset_index()
# Add category type to be plotted lated
top = top.merge(cats[['item_category_id','item_category_name']], how='left', on='item_category_id')

# To print the top selling categories for each year
#print(top)

years = [2013, 2014, 2015]
fig, axes = plt.subplots(1, 3, figsize=(16,6))

#Prepare colors for the top N
colors = [[] for i in range(3)]
for alpha in np.arange(N, 0, -1)/N:
    colors[0].append((.275, .51, .706, alpha))
    colors[1].append((.18, .55, .34, alpha))
    colors[2].append((.5, 0, 0, alpha))
    
for ax, year, cs in zip(axes, years, colors):
    # Get top items for each year
    year_filter = top[top.year==year]
    plot_sizes = year_filter.item_cnt_year
    plot_labels = year_filter.item_category_name.str[:15]#+'('+plot_sizes.astype(str)+')'
    
    # Get the ratio
    print(get_ratio(year, plot_sizes.sum(), N))
    
    # Plot the pie
    ax.pie(plot_sizes, labels=plot_labels, radius=1.5, colors=cs,labeldistance=.5, rotatelabels=True, startangle=90, wedgeprops={"edgecolor":"1",'linewidth': .5})
    # Set titles below pies
    ax.set_title(year, y=-0.2)

# Space pies
fig.tight_layout()
fig.suptitle('Top selling categories for each year', fontsize=16)
plt.show()

del dfp


# * Detecting outliers in item_price and item_cnt_day

# In[6]:


fig, axes = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.5)

flierprops = dict(marker='o', markerfacecolor='cornflowerblue', markersize=6, markeredgecolor='navy')

_ = axes[0].boxplot(x=sales.item_cnt_day, flierprops=flierprops, vert=False)
_ = axes[1].boxplot(x=sales.item_price, flierprops=flierprops, vert=False)

_ = axes[0].set_title('item_cnt_day')
_ = axes[1].set_title('item_price')


# # Data preparation
# * Removing outliers from item_price and item_cnt_day, and duplicate shops:

# In[7]:


sales = sales[(sales.item_price<100000)&(sales.item_price>0)]
sales = sales[(sales.item_cnt_day>0)&(sales.item_cnt_day<1000)]

# Remove duplicate shops
sales.loc[sales.shop_id==0, 'shop_id'] = 57
test.loc[test.shop_id==0, 'shop_id'] = 57

sales.loc[sales.shop_id==1, 'shop_id'] = 58
test.loc[test.shop_id==1, 'shop_id'] = 58

sales.loc[sales.shop_id==10, 'shop_id'] = 11
test.loc[test.shop_id==10, 'shop_id'] = 11


# * Add 'city' and 'category' to shops:\ -The first part of the shop_name is the city e.g.Serguiev Possad \ -The second part of the shop_name is the category e.g. ТЦ (shopping center)

# In[8]:


# Correct the name of a shop
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name"] = 'СергиевПосад ТЦ "7Я"'
# The first part of the shop_name is the city e.g.Serguiev Possad
shops["shop_city"] = shops.shop_name.str.split(' ').map(lambda x: x[0])
# The second part of the shop_name is the category e.g. shopping center
shops["shop_category"] = shops.shop_name.str.split(" ").map(lambda x: x[1])
shops.loc[shops.shop_city == "!Якутск", "shop_city"] = "Якутск" 


# # Feature Engineering

# In[9]:


# Feature encoding
shops["shop_city"] = LabelEncoder().fit_transform(shops.shop_city)
shops["shop_category"] = LabelEncoder().fit_transform(shops.shop_category)
shops = shops[["shop_id", "shop_category", "shop_city"]]
shops.head()


# * Add type and subtype to item_categories:

# In[10]:


item_categories["category_type"] = item_categories.item_category_name.apply(lambda x: x.split(" ")[0]).astype(str)
# The category_type "Gamming" and "accesoires" becomes "Games"
item_categories.loc[(item_categories.category_type=="Игровые")|(item_categories.category_type=="Аксессуары"), "category_type"] = "Игры"
item_categories["split"] = item_categories.item_category_name.apply(lambda x: x.split("-"))
item_categories["category_subtype"] = item_categories.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())


# # Feature Encoding

# In[11]:


item_categories["category_type"] = LabelEncoder().fit_transform(item_categories.category_type)
item_categories["category_subtype"] = LabelEncoder().fit_transform(item_categories.category_subtype)
item_categories = item_categories[["item_category_id", "category_type", "category_subtype"]]
item_categories.head()


# * Compute monthly sales, in the same representation as the test data:

# In[12]:


sales = sales.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)\
          .agg({'item_cnt_day':'sum'})\
          .rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=False)
        
test['date_block_num'] = 34
test['item_cnt_month'] = 0
del test['ID']

df = sales.append(test)
df


# ## Create a feature matrix:

# In[13]:


matrix = []
# Try creating a matrix of product(sales['date_block_num'].unique(), sales.shop_id.unique(), sales.item_id.unique()) which are about 45m lines
for num in df['date_block_num'].unique(): 
    tmp = df[df.date_block_num==num]
    matrix.append(np.array(list(product([num], tmp.shop_id.unique(), tmp.item_id.unique())), dtype='int16'))
    #matrix.append(np.array(list(product([num], shops.shop_id, items.item_id)), dtype='int16'))

# Turn the grid into a dataframe
matrix = pd.DataFrame(np.vstack(matrix), columns=['date_block_num', 'shop_id', 'item_id'], dtype=np.int16)

# Add the features from sales data to the matrix
matrix = matrix.merge(df, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)

#Merge features from shops, items and item_categories:
matrix = matrix.merge(shops, how='left', on='shop_id')
matrix = matrix.merge(items[['item_id','item_category_id']], how='left', on='item_id')
matrix = matrix.merge(item_categories, how='left', on='item_category_id')

# Add month
matrix['month'] = matrix.date_block_num%12
# Clip counts
matrix['item_cnt_month'] = matrix['item_cnt_month'].clip(0, 20)


# In[14]:


# Set columns types to control the matrix' size
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix['month'] = matrix['month'].astype(np.int8)
matrix['item_cnt_month'] = matrix['item_cnt_month'].astype(np.int32)
matrix['shop_category'] = matrix['shop_category'].astype(np.int8)
matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['category_type'] = matrix['category_type'].astype(np.int8)
matrix['category_subtype'] = matrix['category_subtype'].astype(np.int8)
matrix


# In[15]:


print('{0:.2f}'.format(matrix.memory_usage(index=False, deep=True).sum()/(2**20)), 'MB')


# # Feature Engineering
# ### Lagged Features

# In[16]:


def lag_feature(df, lags, col):
    print(col)
    for i in lags:
        shifted = df[["date_block_num", "shop_id", "item_id", col]].copy()
        shifted.columns = ["date_block_num", "shop_id", "item_id", col+"_lag_"+str(i)]
        shifted.date_block_num += i
        df = df.merge(shifted, on=['date_block_num','shop_id','item_id'], how='left').fillna(0)
    return df


# In[17]:


# lag the target item_cnt_month
matrix = lag_feature(matrix, [1, 2, 3, 4, 5, 12], 'item_cnt_month')


# In[18]:


# shop/date_block_num aggregates lags
gb = matrix.groupby(['shop_id', 'date_block_num'],as_index=False)\
          .agg({'item_cnt_month':'sum'})\
          .rename(columns={'item_cnt_month':'cnt_block_shop'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5, 12], 'cnt_block_shop')
matrix.drop('cnt_block_shop', axis=1, inplace=True)


# In[19]:


# item/date_block_num aggregates lags
gb = matrix.groupby(['item_id', 'date_block_num'],as_index=False)\
          .agg({'item_cnt_month':'sum'})\
          .rename(columns={'item_cnt_month':'cnt_block_item'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5, 12], 'cnt_block_item')
matrix.drop('cnt_block_item', axis=1, inplace=True)


# In[20]:


# category/date_block_num aggregates lags
gb = matrix.groupby(['category_type', 'date_block_num'],as_index=False)\
          .agg({'item_cnt_month':'sum'})\
          .rename(columns={'item_cnt_month':'cnt_block_category'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['category_type', 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5, 12], 'cnt_block_category')
matrix.drop('cnt_block_category', axis=1, inplace=True)


# In[21]:


# matrix.to_csv('matrix.csv', index=False)
# matrix = pd.read_csv('matrix.csv')
matrix


# # Label mean encodings
# * Mean encoding and scaling : first split the data into Train and Validation, estimate encodings on Train, then apply them to Validation set:

# In[22]:


from sklearn.preprocessing import StandardScaler

def standard_mean_enc(df, col):
    mean_enc = df.groupby(col).agg({'item_cnt_month': 'mean'})
    scaler = StandardScaler().fit(mean_enc)
    return {v: k[0] for v, k in enumerate(scaler.transform(mean_enc))}


# In[23]:


cols_to_mean_encode = ['shop_category', 'shop_city', 'item_category_id', 'category_type', 'category_subtype']

for col in cols_to_mean_encode:
    # Train on the train data
    mean_enc = standard_mean_enc(matrix[matrix.date_block_num < 33].copy(), col) # X_train, y_train
    # Apply to Train, Validation and Test
    matrix[col] = matrix[col].map(mean_enc)
matrix


# # Splitting Data

# In[24]:


# Remove the 2013's sales data
matrix = matrix[matrix.date_block_num>=12] 
matrix.reset_index(drop=True, inplace=True)
matrix


# In[25]:


X_train = matrix[matrix.date_block_num < 33].drop(['item_cnt_month'], axis=1)
y_train = matrix[matrix.date_block_num < 33]['item_cnt_month']
X_val = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)
y_val =  matrix[matrix.date_block_num == 33]['item_cnt_month']
X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# ### Removing Few Columns

# In[26]:


X_train.drop('date_block_num', axis=1, inplace=True)
X_val.drop('date_block_num', axis=1, inplace=True)
X_test.drop('date_block_num', axis=1, inplace=True)


# # Hyper Parameters Tuning

# In[27]:


splits = []
for block in [27, 28, 29, 30, 31, 32]:
    train_idxs = matrix[matrix.date_block_num < block].index.values
    test_idxs = matrix[matrix.date_block_num == block].index.values
    splits.append((train_idxs, test_idxs))
splits


# In[28]:


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

hyper_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9], 
                'gamma': [0, 0.5, 1, 1.5, 2, 5], 
                'subsample': [0.6, 0.7, 0.8, 0.9, 1], 
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_bin' : [256, 512, 1024]
               }

xgbr = XGBRegressor(seed = 13, tree_method = "hist") #gpu_hist
clf = RandomizedSearchCV(estimator = xgbr, 
                   param_distributions = hyper_params,
                   n_iter = 2, #500
                   scoring = 'neg_root_mean_squared_error',
                   cv = splits,
                   verbose=3)
clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", -clf.best_score_)


# # Model Evaluation
# 
# ## Linear Regression

# In[29]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
yhat_val_lr = lr.predict(X_val).clip(0, 20)
print('Validation RMSE:', mean_squared_error(y_val, yhat_val_lr, squared=False)) #Validation RMSE: 0.9645168655662343
yhat_test_lr = lr.predict(X_test).clip(0, 20)


# # XGBoost

# In[30]:


from xgboost import XGBRegressor

ts = time.time()

xgb = XGBRegressor(seed = 13, 
    tree_method = "hist", #gpu_hist
    subsample = 0.9,
    max_depth = 9,
    learning_rate = 0.1,
    gamma = 2,
    colsample_bytree = 0.9
    )
xgb.fit(
    X_train,y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
    early_stopping_rounds = 10
    )
print('Training took: {0}s'.format(time.time()-ts))
yhat_val_xgb = xgb.predict(X_val).clip(0, 20)
print('Valdation RMSE:', mean_squared_error(y_val, yhat_val_xgb, squared=False)) #Valdation RMSE: 0.9273184120626018
yhat_test_xgb = xgb.predict(X_test).clip(0, 20)


# * Serialize and Deserialize the XGBoost model with Pickle:

# # Importing Pickel

# In[31]:


import pickle
pickle.dump(xgb, open("xgboost.pickle.dat", "wb"))
#loaded_model = pickle.load(open("xgboost_base.pickle.dat", "rb"))


# # Ploting XGBoost Feaure Importance

# In[32]:


from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(xgb, (10,14))


# # Esembling
# ## Meta Features

# In[33]:


y_train_meta = matrix[matrix.date_block_num.isin([27, 28, 29, 30, 31, 32])].item_cnt_month


# In[34]:


X_train_meta = [[],[]]
for block in [27, 28, 29, 30, 31, 32]:
    print('Block:', block)
    # X and y Train for blocks from 12 to block
    X_train_block = matrix[matrix.date_block_num < block].drop(['date_block_num', 'item_cnt_month'], axis=1)
    y_train_block = matrix[matrix.date_block_num < block].item_cnt_month
    # X and y Test for block
    X_val_block = matrix[matrix.date_block_num == block].drop(['date_block_num', 'item_cnt_month'], axis=1)
    #y_test_block = matrix[matrix.date_block_num == block].item_cnt_month
    
    # Fit first model 
    print(' LR fitting ...')
    lr.fit(X_train_block, y_train_block)
    print(' LR fitting ... done')
    # Append prediction results on X_val_block to X_train_meta (first column)
    X_train_meta[0] += list(lr.predict(X_val_block).clip(0, 20))
    
    # Fit second model
    print(' XGB fitting ...')
    xgb.fit(
        X_train_block, y_train_block,
        eval_metric="rmse",
        eval_set=[(X_train_block, y_train_block)],
        #eval_set=[(X_train_block, y_train_block), (X_val_block, y_test_block)],
        verbose=0,
        early_stopping_rounds = 10
    )
    print(' XGB fitting ... done')
    # Append prediction results on X_val_block to X_train_meta (second column)
    X_train_meta[1] += list(xgb.predict(X_val_block).clip(0, 20))
# Turn list into dataframe
X_train_meta = pd.DataFrame({'yhat_lr': X_train_meta[0], 'yhat_xgb': X_train_meta[1]})


# In[35]:


plt.scatter(X_train_meta.yhat_lr, X_train_meta.yhat_xgb)


# # Stacking

# In[36]:


stacking = LinearRegression()
stacking.fit(X_train_meta, y_train_meta)

#Squared: If True returns MSE value, if False returns RMSE value.
yhat_train_meta = stacking.predict(X_train_meta).clip(0, 20)
print('Meta Training RMSE:', mean_squared_error(y_train_meta, yhat_train_meta, squared=False))
# Meta Training RMSE: 0.813971713370181

yhat_val_meta = stacking.predict(np.vstack((yhat_val_lr, yhat_val_xgb)).T).clip(0, 20)
print('Meta Validation RMSE:', mean_squared_error(y_val, yhat_val_meta, squared=False))
# Meta Validation RMSE: 0.9184725317670576

yhat_test_meta = stacking.predict(np.vstack((yhat_test_lr, yhat_test_xgb)).T).clip(0, 20)


# # Submission

# In[37]:


submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": yhat_test_meta
})
submission.to_csv('submission_stacking.csv', index=False)
# Public score 0.92466


# In[ ]:




