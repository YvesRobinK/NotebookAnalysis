#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import tqdm
from sklearn.impute import KNNImputer
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder
import joblib
from sklearn.metrics import mean_squared_error


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
#for dirname, _, filenames in os.walk('../data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading Data

# In[2]:


train = pd.read_csv('../input/widsdatathon2022/train.csv')
#train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../input/widsdatathon2022/test.csv')
#test = pd.read_csv('../data/test.csv')

#we will suppose that two  if two buildings have the same values for these features ; they are the sane building
# in other words groupby_cols = (building_id)
groupby_cols = ['State_Factor','building_class','facility_type','floor_area','year_built']
train['source'] = 'train'
test['source']  = 'test'
df = pd.concat([train,test], 0, ignore_index=True)
df=df.sort_values(by=groupby_cols+['Year_Factor']).reset_index(drop=True)


# In[3]:


df.index


# ### Missing values imputation

# ### Category encoding
# As we donwe will use **knn imputing**

# In[4]:


# cats
df.loc[:,df.dtypes=='object'].columns


# In[5]:


facility_type_qcut = pd.qcut(df['facility_type'].value_counts(), q=4, labels=[f'facility_type_qcut_{i}' for i in range(4)])
map_dict = dict(facility_type_qcut)
for i in map_dict:
    if map_dict[i] == 'facility_type_qcut_3':
        map_dict[i] = i
df['facility_type'] = df['facility_type'].map(map_dict)


# In[6]:


cats = ['State_Factor', 'facility_type', 'building_class']
for col in cats:
    dummies = pd.get_dummies(df[col], dummy_na=False)
    for ohe_col in dummies:
        df[f'ohe_{col}_{ohe_col}'] = dummies[ohe_col]


# In[7]:


# Function to calculate missing values by column# Funct 
# from https://www.kaggle.com/parulpandey/starter-code-with-baseline
def missing_values_table(df):
        # Total missing values by column
        mis_val = df.isnull().sum()
        
        # Percentage of missing values by column
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # build a table with the thw columns
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Missing values for training data
missing_values_train = missing_values_table(train)
missing_values_train[:20].style.background_gradient(cmap='Reds')


# In[8]:


knn_imputing = False
target='site_eui'

if knn_imputing:
    imputer = KNNImputer(n_neighbors=7)
    tmp = df[['State_Factor', 'building_class', 'facility_type', 'source', target]]
    df = df.drop(tmp.columns, axis=1)
    df1 = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

    joblib.dump(imputer, '../models/knn_imputer.pkl')

    for col in tmp.columns:
        df[col]=tmp[col]
    for col in df1.columns:
        df[col] = df1[col]


# ### Target Encoding

# In[9]:


cats = ['State_Factor', 'building_class', 'facility_type']
for col in cats:
    encoder = TargetEncoder()
    df[f'te_{col}'] = encoder.fit_transform(df[col], df[target])


# ## Weather based features
# 
# we will extract new weather statistics from the building location weather features 

# In[10]:


# extract new weather statistics from the building location weather features
temp = [col for col in df.columns if 'temp' in col]

df['min_temp'] = df[temp].min(axis=1)
df['max_temp'] = df[temp].max(axis=1)
df['avg_temp'] = df[temp].mean(axis=1)
df['std_temp'] = df[temp].std(axis=1)
df['skew_temp'] = df[temp].skew(axis=1)

# by seasons
temp = pd.Series([col for col in df.columns if 'temp' in col])

winter_temp = temp[temp.apply(lambda x: ('january' in x or 'february' in x or 'december' in x))].values
spring_temp = temp[temp.apply(lambda x: ('march' in x or 'april' in x or 'may' in x))].values
summer_temp = temp[temp.apply(lambda x: ('june' in x or 'july' in x or 'august' in x))].values
autumn_temp = temp[temp.apply(lambda x: ('september' in x or 'october' in x or 'november' in x))].values


### winter
df['min_winter_temp'] = df[winter_temp].min(axis=1)
df['max_winter_temp'] = df[winter_temp].max(axis=1)
df['avg_winter_temp'] = df[winter_temp].mean(axis=1)
df['std_winter_temp'] = df[winter_temp].std(axis=1)
df['skew_winter_temp'] = df[winter_temp].skew(axis=1)
### spring
df['min_spring_temp'] = df[spring_temp].min(axis=1)
df['max_spring_temp'] = df[spring_temp].max(axis=1)
df['avg_spring_temp'] = df[spring_temp].mean(axis=1)
df['std_spring_temp'] = df[spring_temp].std(axis=1)
df['skew_spring_temp'] = df[spring_temp].skew(axis=1)
### summer
df['min_summer_temp'] = df[summer_temp].min(axis=1)
df['max_summer_temp'] = df[summer_temp].max(axis=1)
df['avg_summer_temp'] = df[summer_temp].mean(axis=1)
df['std_summer_temp'] = df[summer_temp].max(axis=1)
df['skew_summer_temp'] = df[summer_temp].max(axis=1)
## autumn
df['min_autumn_temp'] = df[autumn_temp].min(axis=1)
df['max_autumn_temp'] = df[autumn_temp].max(axis=1)
df['avg_autumn_temp'] = df[autumn_temp].mean(axis=1)
df['std_autumn_temp'] = df[autumn_temp].std(axis=1)
df['skew_autumn_temp'] = df[autumn_temp].skew(axis=1)


# In[11]:


df['month_cooling_degree_days'] = df['cooling_degree_days']/12
df['month_heating_degree_days'] = df['heating_degree_days']/12


# ### Buildig based feature:

# In[12]:


# total area
df['building_area'] = df['floor_area'] * df['ELEVATION']
# rating energy by floor
df['floor_energy_star_rating'] = df['energy_star_rating']/df['ELEVATION']


# ## Lag based features:
# we will compute lagged features values up to three years for the following features:
# * site_eui
# * energy_star
# * ELEVATION
# * temp features

# In[13]:


temp_features = [col for col in df.columns if 'temp' in col]

for lag_year in [1,2,3] :
    # building mesh
    df[f'site_eui_lag{lag_year}'] = df.groupby(groupby_cols)['site_eui'].shift(lag_year) # because it is sorted by year factor
    df[f'energy_star_rating_lag{lag_year}'] = df.groupby(groupby_cols)['energy_star_rating'].shift(lag_year)
    df[f'ELEVATION_lag{lag_year}'] = df.groupby(groupby_cols)['ELEVATION'].shift(lag_year)
    for temp in temp_features:
        df[f'{temp}_lag{lag_year}'] = df.groupby(groupby_cols)[temp].shift(lag_year)


# ### Rolling based features :
# average of feature value through the last 3 years

# In[14]:


# rolling average
for i in range(1, 3) :
    # building mesh
    df[f'site_eui_roll_avg{i}'] = df.groupby(groupby_cols)['site_eui'].shift(i).rolling(3).mean()
    df[f'energy_star_rating_roll_avg{i}'] = df.groupby(groupby_cols)['energy_star_rating'].shift(i).rolling(3).mean()
    df[f'ELEVATION_roll_avg{i}'] = df.groupby(groupby_cols)['ELEVATION'].shift(i).rolling(3).mean()
    for temp in temp_features:
        df[f'{temp}_roll_avg{i}'] = df.groupby(groupby_cols)[temp].shift(i).rolling(3).mean()


# ## Delta based features
# Delta current value vs previous year:
# 

# In[15]:


for var in ['site_eui', 'energy_star_rating', 'ELEVATION'] + temp_features:
    df[f'delta_{var}_2_1']  = np.where(df[f'{var}_lag2'].notnull() & df[f'{var}_lag1'].notnull() ,
                                    abs(df[f'{var}_lag1']-df[f'{var}_lag2'] ) / df[f'{var}_lag2'],
                                    np.nan )
    # df[f'{var}_lag1'] = df[f'{var}_lag2'] + df[f'{var}_lag2'] * df[f'delta_{var}_2_1']
    # we consider df[f'{var}'] = df[f'{var}_lag1'] + df[f'{var}_lag1'] * df[f'delta_{var}_2_1']
    df[f'{var}_from_delta2_1']  = np.where(df[f'{var}_lag1'].notnull() ,
                                    df[f'{var}_lag1'] + df[f'{var}_lag1'] * df[f'delta_{var}_2_1'],
                                    np.nan )


# In[16]:


df['site_eui_coeff'] = (df['energy_star_rating_lag2']* df[f'site_eui_lag1'] - df[f'site_eui_lag2']*df['energy_star_rating_lag2'])/(df['site_eui_lag1'] - df['site_eui_lag2'])
df['site_eui_from_site_eui_coeff'] = df['site_eui_coeff']*df[f'site_eui_lag1']



# In[17]:


# groupby function
for col in ['State_Factor', 'building_class', 'facility_type', 'energy_star_rating']:
    print(col)
    cols=[col, 'Year_Factor']
    tmp = df.sort_values(cols).groupby(cols).agg({
                "site_eui": ['mean', 'median','max','min','sum']  ,
    #            "year_built": ['mean', 'median','max','min'] ,
                "energy_star_rating" : ['min','max','mean', 'median'] ,
                "id" : "count"
                          }).shift(1).reset_index()
    for idx in tmp.columns:
        if idx[1] != '':
            print(f'{idx[1]}_{idx[0]}_by_{col}_lag1')
            cols.append(f'{idx[1]}_{idx[0]}_by_{col}_lag1')
    tmp.columns = cols
    df = df.merge(tmp, on=[col, 'Year_Factor'], how='left')
    print(df.shape)


# In[18]:


target = 'site_eui'
plt.figure(figsize=(10,7))
# plot the original variable vs sale price    
plt.subplot(2, 1, 1)
train[target].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Original ' + target)

# plot transformed variable vs sale price
plt.subplot(2, 1, 2)
np.log(train[target]).hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Transformed ' + target)



# In[19]:


train.dtypes != 'object'


# In[20]:


nums = train.loc[:, train.dtypes != 'object'].columns
df[nums].hist(bins=50, figsize=(20,20))
plt.show()


# In[21]:


df[nums].skew().sort_values(key=abs, ascending=False)[:5]


# Binarize very skewed variables

# In[22]:


skewed = [
    'days_above_110F', 'days_above_100F'
]

for var in skewed:
    
    # map the variable values into 0 and 1
    df[var] = np.where(df[var]==0, 0, 1)


# In[23]:


# save data
saved = True
if saved:
    get_ipython().system('pip3 install pickle5')
    import pickle5 as pickle
    data_path = '../input/feature-engineering-wids2022/feature_engineering.pkl'
    with open(data_path, "rb") as fh:
        df = pickle.load(fh)
else:
    df.to_pickle('../feature_engineering.pkl')


# In[ ]:





# In[24]:


# later

df.shape


# In[25]:


df.head()


# ## Catboost

# In[26]:


train = df[df['source']=='train']
test = df[df['source']=='train']

cats = ['State_Factor', 'facility_type', 'building_class']
    


# In[ ]:





# In[27]:


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test_ids = test['id']
train_ids = train['id']
target = train['site_eui']
train = train.drop(['id', 'source', 'site_eui'], axis=1)#['id', 'source', 'Year_Factor', target]+cats
test = test.drop(['id', 'source', 'site_eui'], axis=1)

# get discrete end categorical features colums indexes 
# needed later for the cat bosst model
cats_discrete_idx = np.where(train.dtypes != 'float64')[0]
# create the label
le = LabelEncoder()
for col_idx in cats_discrete_idx:
    train.iloc[:, col_idx] = le.fit_transform(train.iloc[:, col_idx].astype(str))
    test.iloc[:, col_idx] = le.transform(test.iloc[:, col_idx].astype(str))


# In[28]:


for i in cats_discrete_idx:
    print(train.columns[i])


# In[29]:


# prepaere the out of folds predictions 
train_oof = np.zeros((train.shape[0],))
test_preds = np.zeros(test.shape[0])


# In[30]:


NUM_FOLDS = 10
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
# we. can also use 

for fold, (train_idx, test_idx) in tqdm.tqdm(enumerate(kf.split(train, target))):
    X_train, X_test = train.iloc[train_idx][test.columns], train.iloc[test_idx][test.columns]
    y_train, y_test = target[train_idx], target[test_idx]
    
    ## config from https://www.kaggle.com/nicapotato/simple-catboost
    model = CatBoostRegressor(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='RMSE',
#                         early_stopping_rounds=42,
                         random_seed = 23,
                         bagging_temperature = 0.2,
                         od_type='Iter',
                         metric_period = 75,
                         od_wait=100)
    # train model
    model.fit(X_train, y_train,
                 eval_set=(X_test,y_test),
                 cat_features=cats_discrete_idx,
                 use_best_model=True,
                 verbose=True)

    oof = model.predict(X_test)
    train_oof[test_idx] = oof
    test_preds += model.predict(test)/NUM_FOLDS      
    print(f"out-of-folds prdiction ==== fold_{fold} RMSE",np.sqrt(mean_squared_error(oof, y_test, squared=False)))


# ## Save Data

# In[31]:


# save results
np.save('train_oof.npy', train_oof)
np.save('test_preds.npy', test_preds)


# In[32]:


sub = pd.DataFrame(columns=['id', 'site_eui'])
sub['id']  = test_ids
sub['site_eui'] = test_preds
sub.to_csv('submission_CB2.csv', index=False)


# In[ ]:




