#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# 
# 
# **In this notebook, I have built sales prediction models using sci-kit learn and statsmodels libraries to predict sales price of store sales data.**
# 
# **I have mentioned various regression metrics to measure performance of regression models with their applications.**

# # Table of contents:
# 
# 1) **[Descriptive and exploratory analysis](#dea)**
# 
# 2) **[Train and test data preparation](#dataprep)**
# 
# 3) **[Feature engineering](#fe)**
# 
# 4) **[Model development and evaluation](#model)**
# 
# **[Residual analysis](#resid)**
# 
# **[Test of Homoscedasticity](#test)**
# 
# **[Outlier analysis](#out)**
# 
# **[Leverage values](#lv)**

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from xgboost import XGBRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df_sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
df_shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
df_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
df_sub = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')


def df_head(df):
    return df.head()

    
df_head(df_sales)


# In[3]:


df_head(df_sub)


# In[4]:


df_head(df_items)


# In[5]:


df_head(df_shops)


# In[6]:


df_head(df_test)


# # Descriptive and exploratory data analysis
# 
# <a id="dea"></a>

# In[7]:


df_sales.describe().T


# In[8]:


df_test.describe().T


# In[9]:


df_sales['item_price']


# In[10]:


df_sales[['shop_id','item_id','item_price','item_cnt_day']].corr()


# In[11]:


df_items.describe().T


# In[12]:


plt.figure(figsize=(8,5))
plt.hist(df_sales['item_id'])
plt.show


# > Distribution of item id

# In[13]:


plt.figure(figsize=(8,5))
plt.hist(df_items['item_category_id'])
plt.show


# > Distribution of item categories id

# In[14]:


plt.figure(figsize=(8,5))
plt.hist(df_sales['item_cnt_day'])
plt.show


# In[15]:


df_items.groupby('item_category_id').count()


# In[16]:


df_items.groupby('item_category_id').mean()


# In[17]:


df_items['diff_col_of_item_id'] = df_items.groupby('item_category_id')['item_id'].max() - df_items.groupby('item_category_id')['item_id'].min()

df_items.head()


# In[18]:


#df_items.drop('diff_col', inplace=True, axis=1)
#df_items


# In[19]:


df_items.head()


# > What we have found so far:
# 
# 1. item id and shop id will be only independent variable will predict target variable
# 2. we will drop item price column from train data set
# 3. shop ids are between 1 t0 60
# 4. item id and item price are correlate with each other
# 5. each item id fall into certain item category as item ids >> item category
# 6. we can assign new column item_category to each item id

# # Train and test data set preparation and pre-processing
# 
# <a id="dataprep"></a>

# In[20]:


df_sales.head()


# In[21]:


df_sales.isnull().sum()


# In[22]:


df_sales.drop_duplicates(keep='first', inplace=True, ignore_index=True)

df_sales.head()


# In[23]:


df_sales[df_sales['item_price'] <0]


# In[24]:


df_sales.drop(df_sales[df_sales['item_cnt_day'] <0].index , inplace=True)
df_sales.drop(df_sales[df_sales['item_price'] <0].index , inplace=True)

df_sales.shape


# ## outliers removal

# In[25]:


Q1 = np.percentile(df_sales['item_price'], 25.0)
Q3 = np.percentile(df_sales['item_price'], 75.0)

IQR = Q3 - Q1

df_sub1 = df_sales[df_sales['item_price'] > Q3 + 1.5*IQR]
df_sub2 = df_sales[df_sales['item_price'] < Q1 - 1.5*IQR]

df_sales.drop(df_sub1.index, inplace=True)

df_sales.shape


# In[26]:


df_sales['date_block_num'].unique()


# In[27]:


df_sales.groupby('date_block_num')['item_id'].mean()


# In[28]:


price = round(np.array(df_sales.groupby('date_block_num')['item_price'].mean()).mean(),2)
print(price)


# In[29]:


dict(round(df_sales.groupby('date_block_num')['item_price'].mean(),4))


# In[30]:


df_sales.head()


# In[31]:


df_test.head()


# # Feature engineering
# 
# <a id="fe"></a>

# #### FE workflow

# > create columns with mean price by date block in train and test dataset, remove item price from train
# 
# > remove data_ block column from train data set
# 
# > create new cloumns with mean price per shop id for both train and test dataset
# 
# > merge df_items table with train and test dataset on item id and create new column item category
# 

# In[32]:


#df_sales.drop('mean_price_data_block', inplace=True, axis=1)

replace_dict = dict(round(df_sales.groupby('date_block_num')['item_price'].mean(),2))


# In[33]:


df_sales['date_block_num'] = df_sales['date_block_num'].replace(replace_dict)

df_train = df_sales.copy()
df_train.drop(['date','item_price'], axis=1, inplace=True)
df_train.rename(columns = {'date_block_num':'mean_price_by_column'}, inplace=True)
df_train.head()


# In[34]:


mean_price = np.array(df_sales.groupby('date_block_num')['item_price'].mean()).mean()
mean_price


# In[35]:


df_test.shape


# In[36]:


df_train.shape


# In[37]:


#df_test.drop('ID', inplace=True, axis=1)
df_test.head()
com_df = pd.concat([df_train,df_test])

com_df['mean_price_by_column'] = com_df['mean_price_by_column'].fillna(value=price)
com_df['item_cnt_day'] = com_df['item_cnt_day'].fillna(value=0)

test_df = com_df[com_df['item_cnt_day'] == 0]
train_df = com_df[com_df['item_cnt_day'] != 0]


# In[38]:


test_df.shape


# In[39]:


testdf = test_df.copy()

testdf.drop('ID', inplace=True, axis=1)
testdf.drop('item_cnt_day', inplace=True, axis=1)
testdf


# In[40]:


traindf = train_df.copy()

traindf.drop('ID', inplace=True, axis=1)


# In[41]:


traindf.head()


# # Train data and test data for modelling and evalution
# 
# <a id="model"></a>

# In[42]:


#test_df.drop('item_cnt_day', inplace=True, axis=1)
testdf['item_id'] = (testdf['item_id'] - testdf['item_id'].mean())/testdf['item_id'].std()
testdf.head()


# In[43]:


traindf['item_id'] = (traindf['item_id'] - traindf['item_id'].mean())/traindf['item_id'].std()
traindf.head()


# # Model 1 

# In[44]:


X = traindf.loc[:,['mean_price_by_column','shop_id','item_id']]
y = traindf.loc[:,'item_cnt_day']


# In[45]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.8, random_state= 42)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print("regression coefficients are: " , model1.coef_)

y_pred = model1.predict(X_valid)


MSE = mean_squared_error(y_valid,y_pred)
MAE = mean_absolute_error(y_valid,y_pred)
R2  = r2_score(y_valid,y_pred)

print("MSE: ", MSE)
print("MAE: ", MAE)
print("R2: ", R2)


# In[46]:


#model_sub_pred = model1.predict(testdf)

#sub_df = pd.DataFrame(data= {'ID':np.array(df_test['ID']),'item_cnt_month':model_sub_pred})

#sub_df.to_csv('submission.csv', index=False)


# # Model 2

# In[47]:


# addding constant as statsmodels api does not include it!
X_new = X.copy()
X_new = sm.add_constant(X_new)
test_df_new = test_df.copy()
test_df_new = sm.add_constant(test_df_new)
X_train_new, X_valid_new, y_train_new, y_valid_new = train_test_split(X_new, y,train_size=0.8, random_state= 42)


# In[48]:


#using statsmodel api

stats_model = sm.OLS(y_train_new, X_train_new)
stat_fit = stats_model.fit()
print("Model coeffieciants: ", stat_fit.params)

print("\nModel summary: ", stat_fit.summary2())


# > P value is less than 0.05 indicates that model is statistically significant
# 
# > F-stat is also very low
# 
# > R2 is almost 17%, that explains independent variables explain 17% of variation in dependent variable item count

# ## Residual analysis
# 
# <a id="resid"></a>

# In[49]:


model_residual = stat_fit.resid
probplot = sm.ProbPlot(model_residual)
plt.figure(figsize=(8,8))
probplot.ppplot(line = '45')
plt.title("Rsidual analysis of model 2")
plt.show()


# > theoritical probabilties and sample probabilities are not corellating with each other, it implies that resdiuls are not folloewing normal distribution.
# 
# > model is not a good fit to data as data is polynomial.

# ## Test of Homoscedasticity
# 
# <a id="test"></a>

# In[50]:


def get_std_val(vals):
    return (vals - vals.mean())/vals.std()

plt.figure(figsize=(8,8))
plt.scatter(get_std_val(stat_fit.fittedvalues), get_std_val(model_residual))
plt.title("Residual plot")
plt.xlabel("predicted values")
plt.ylabel("Residuals")
plt.show()


# > There is a clear funnel shape is observed and we can see that there is couple of outliers in actual values of y.

# ## Outlier analysis
# 
# <a id="out"></a>

# In[51]:


traindf['z_score_item'] = zscore(traindf['item_cnt_day'])


# In[52]:


#outliears in y variable

traindf[(traindf['z_score_item'] > 3.0) | (traindf['z_score_item'] < -3.0)]


# > Total 9971 rows are having extream item sales which resulted in higher residuals

# ## Cook's distance
# 

# In[53]:


#item_influence = stat_fit.get_influence()
#(c, p) = item_influence.cooks_distance

#plt.stem(np.arange(len(X_train)),
   #      np.round(c, 3),
     #    markerfmt=',')
#plt.title("Cook's distance")
#plt.xlabel("row index")
#plt.ylabel('Cook\'s distance')
#plt.show


# > Using influence method we can plot cooks distance to find which observance has a most influence on output variable

# ## Leverage Values
# 
# <a id='lv'></a>

# In[54]:


#fig, ax = plt.subplots(figsize=(8,6))
#influence_plot(stat_fit, ax=ax)
#plt.title("Influence plot")
#plt.show()


# In[55]:


pred_y = stat_fit.predict(X_valid_new)

r2 = r2_score(y_valid_new,pred_y)
mse = mean_squared_error(y_valid_new,pred_y)
mae = mean_absolute_error(y_valid_new,pred_y)

print(r2)
print(mse)
print(mae)


# In[56]:


#model_sub = stat_fit.predict(testdf)

#sub_dff = pd.DataFrame(data= {'ID':np.array(df_test['ID']),'item_cnt_month':model_sub})

#sub_dff.to_csv('submission.csv', index=False) 


# ## Calculating prediction intervals

# In[57]:


pred_y = stat_fit.predict(X_valid_new)

_, pred_y_low, pred_y_high = wls_prediction_std( stat_fit, 
                                                X_valid_new, 
                                                alpha = 0.1)

pred_int_df = pd.DataFrame({'item_id_z': X_valid['item_id'],
                            'pred_y': np.array(pred_y),
                            'Pred_y_low': pred_y_low,
                             'Pred_y_high': pred_y_high
                           })

pred_int_df.head(10)


# > Using statsmodels wls_prediction_std method we have calculated prediction interval for each predicted  value of y.

# # Model 3

# In[58]:


model3 = XGBRegressor(n_estimators=50,
                      max_depth=3,
                      learning_rate = 0.01)

model3.fit(X_train, y_train)

prey = model3.predict(X_valid)

sq_error = mean_squared_error(y_valid, prey)

print(sq_error)


# In[59]:


model3_sub = model3.predict(testdf)

sub_dff2 = pd.DataFrame(data= {'ID':np.array(df_test['ID']),'item_cnt_month':model3_sub})

sub_dff2.to_csv('submission.csv', index=False) 


# ## **If you liked this notebook, Do upvote and share your feedback on the same**
