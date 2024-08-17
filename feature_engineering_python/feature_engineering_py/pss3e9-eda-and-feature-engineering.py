#!/usr/bin/env python
# coding: utf-8

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Concrete Strength Prediction</p>
# Regression task to predict concrete strength

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import math

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()

rc = {
    "axes.facecolor": "#FFF9ED",
    "figure.facecolor": "#FFF9ED",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Loading Data</p>

# In[3]:


train_set = pd.read_csv('/kaggle/input/playground-series-s3e9/train.csv')
test_set = pd.read_csv('/kaggle/input/playground-series-s3e9/test.csv')
concrete_set = pd.read_csv('/kaggle/input/predict-concrete-strength/ConcreteStrengthData.csv')

train_set['Generated'] = test_set['Generated'] = 1
concrete_set['Generated'] = 0


# Description of Fields are as follows:-
# 
# 1. **CementComponent**:- Amount of cement is mixed
# 2. **BlastFurnaceSlag**:- Amount of Blast Furnace Slag is mixed
# 3. **FlyAshComponent**:- Amount of FlyAsh is mixed
# 4. **WaterComponent**:- Amount of water is mixed
# 5. **SuperplasticizerComponent**:- Amount of Super plasticizer is mixed
# 6. **CoarseAggregateComponent**:- Amount of Coarse Aggregate is mixed
# 7. **FineAggregateComponent**:- Amount of Coarse Aggregate is mixed
# 8. **AgeInDays**:- How many days it was left dry
# 9. **Strength**:- What was the final strength of concrete- (Target)

# In[4]:


print(train_set.info(),'\n')
print(test_set.info(),'\n')
print(concrete_set.info(),'\n')


# In[5]:


train_set.drop('id',axis=1,inplace=True)
concrete_set.columns = train_set.columns # Fix the columns


# In[6]:


full_set = pd.concat([train_set,concrete_set],axis=0)
full_set


# In[7]:


full_set.drop_duplicates(inplace=True)


# In[8]:


full_set.info()


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Exploratory Data Analysis</p>
# **Checking features distribution**

# In[9]:


from scipy.stats import probplot

features = list(full_set.columns)
fig, axes = plt.subplots(len(features),3,figsize=(20,45))

for f,(ax1,ax2,ax3) in enumerate(axes):
    sns.distplot(full_set[features[f]],ax=ax1,hist_kws={'color':'orange'}).lines[0].set_color('black')
    sns.boxplot(x=features[f],data=full_set,ax=ax2,palette='cubehelix')
    probplot(full_set[features[f]],plot=ax3)

fig.suptitle('Data Distribution in Training Set',fontsize=22,fontweight='bold',y=0.9)
plt.show()


# In[10]:


def winsorize(df, column, upper, lower):
    col_df = df[column]
    
    perc_upper = np.percentile(df[column],upper)
    perc_lower = np.percentile(df[column],lower)
    
    df[column] = np.where(df[column] >= perc_upper, 
                          perc_upper, 
                          df[column])
    
    df[column] = np.where(df[column] <= perc_lower, 
                          perc_lower, 
                          df[column])
    
    return df


# In[11]:


full1 = full_set.copy()
for col in features:
    if col in ['SuperplasticizerComponent','AgeInDays','WaterComponent']:
        full1 = winsorize(full1,col,97.5,0.025)


# In[12]:


features = set(full_set.columns).difference({'Generated','Strength'})
features = list(features)
fig, axes = plt.subplots(len(features)//3+1,3,figsize=(20,15))

f = 0
for axs in axes:
    for ax in axs:
        if f == len(features): break
        sns.regplot(x=features[f],y='Strength',data=full_set,color='#5a2815',line_kws={'color':'orange'},ax=ax)
        f += 1

fig.suptitle('Direct Relationship of Each Features and Target',y=0.95,fontsize=22,fontweight='bold')        
plt.show()


# In[13]:


fig, axes = plt.subplots(len(features)//3+1,3,figsize=(20,15))

f = 0
for axs in axes:
    for ax in axs:
        if f == len(features): break
        sns.boxplot(x=features[f],y='Strength',data=full_set,ax=ax)
        f += 1

fig.suptitle('Direct Relationship of Each Features and Target',y=0.95,fontsize=22,fontweight='bold')        
plt.show()


# We do not see a distinct linear relationship between our features and target except for `AgeInDays`. So it is most likely that non-linear models or tree-based models will perform better for our data.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Feature Engineering</p>
# 
# Concrete strength is affected by many factors, such as quality of raw materials, water/cement ratio, coarse/fine aggregate ratio, age of concrete, compaction of concrete, temperature, relative humidity and curing of concrete.
# 
# The feature engineering ideas are inspired from [this website](https://theconstructor.org/concrete/factors-affecting-strength-of-concrete/6220/) and [this website](https://www.structuralguide.com/factors-affecting-strength-of-concrete/)

# In[14]:


full_set['cement-water ratio'] = full_set['CementComponent'] / full_set['WaterComponent']
full_set['aggregate-cement ratio'] = (full_set['FineAggregateComponent'] + full_set['CoarseAggregateComponent']) / full_set['CementComponent']
full_set['age_cement'] = full_set['CementComponent'] / full_set['AgeInDays']
full_set['superplasticizer-cement ratio'] = full_set['SuperplasticizerComponent'] / full_set['CementComponent']

full1['cement-water ratio'] = full1['CementComponent'] / full1['WaterComponent']
full1['aggregate-cement ratio'] = (full1['FineAggregateComponent'] + full1['CoarseAggregateComponent']) / full1['CementComponent']
full1['age_cement'] = full1['CementComponent'] / full1['AgeInDays']
full1['superplasticizer-cement ratio'] = full1['SuperplasticizerComponent'] / full1['CementComponent']

test_set['cement-water ratio'] = (test_set['CementComponent'] / test_set['WaterComponent'])
test_set['aggregate-cement ratio'] = (test_set['FineAggregateComponent'] + test_set['CoarseAggregateComponent']) / test_set['CementComponent']
test_set['age_cement'] = test_set['AgeInDays'] * test_set['CementComponent']
test_set['superplasticizer-cement ratio'] = test_set['SuperplasticizerComponent'] / test_set['CementComponent']


# In[15]:


features = set(full_set.columns).difference({'Generated','Strength','Strength_boxcox'})
features = list(features)
fig, axes = plt.subplots(len(features)//3,3,figsize=(20,25))

f = 0
for axs in axes:
    for ax in axs:
        if f == len(features): break
        sns.distplot(full_set[features[f]],ax=ax,label='train')
        sns.distplot(full1[features[f]],ax=ax,label='train (winsorized)')
        sns.distplot(test_set[features[f]],ax=ax,label='test').lines[0].set_color('crimson')
        ax.legend()
        f += 1
        
fig.suptitle('Data Distribution',fontsize=22,fontweight='bold',y=0.95)
plt.show()


# In[16]:


def heatmap_corr(df,title):    
    plt.figure(figsize=(20,10))
    plt.title(title,fontsize=20,fontweight='bold')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr,dtype=bool))
    sns.heatmap(corr,mask=mask,annot=True,fmt='.2f')
    plt.xticks(ticks=range(len(df.columns)),labels=df.columns,rotation=45)


# In[17]:


heatmap_corr(full_set,'Training Data Correlation')
heatmap_corr(full1,'Training Data (Winsorized) Correlation')
heatmap_corr(test_set,'Testing Data Correlation')


# In[18]:


drop = ['Strength','CementComponent']
X = full1.drop(drop,axis=1)
y = full1['Strength']


# In[19]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
X_scaled = X.copy()
X_scaled[X.columns] = rs.fit_transform(X)
test_scaled = rs.transform(test_set[X.columns])

kf = KFold(n_splits=5,shuffle=True,random_state=123)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Modelling and Evaluation</p>

# In[20]:


from xgboost import XGBRegressor

xgb_params = {
    'n_estimators':250,
    'max_depth':3,
    'learning_rate':0.07
}

rmse = []
for fold, (train_idx,test_idx) in enumerate(kf.split(X,y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train,y_train)
    
    y_pred = xgb.predict(X_test)
    err = np.sqrt(mse(y_test,y_pred))
    rmse.append(err)
    print(f'FOLD {fold+1} ===== RMSE : {err}')

print(f'ALL FOLDS RMSE : {np.mean(rmse)}')


# In[21]:


import xgboost
xgboost.plot_importance(xgb, title='Feature Importance', xlabel='Importance Score', ylabel='Feature');


# In[22]:


from lightgbm import LGBMRegressor

lgb_params = {
    'learning_rate':0.05,
    'n_estimators':300,
    'max_depth':3,
    'colsample_bytree':0.96
}

rmse = []
for fold, (train_idx,test_idx) in enumerate(kf.split(X,y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    lgb = LGBMRegressor(**lgb_params)
    lgb.fit(X_train,y_train)
    
    y_pred = lgb.predict(X_test)
    err = np.sqrt(mse(y_test,y_pred))
    rmse.append(err)
    print(f'FOLD {fold+1} ===== RMSE : {err}')

print(f'ALL FOLDS RMSE : {np.mean(rmse)}')


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Making Submission</p>

# In[23]:


xgb = XGBRegressor(**xgb_params)
lgb = LGBMRegressor(**lgb_params)

xgb.fit(X,y)
lgb.fit(X,y)

xgb_pred = xgb.predict(test_set[X.columns])
lgb_pred = lgb.predict(test_set[X.columns])

plt.figure(figsize=(10,4))
sns.distplot(lgb_pred,label='lgb pred')
sns.distplot(xgb_pred,label='xgb pred')
plt.legend();


# In[24]:


# submission = test_set[['id']].copy()
# submission['Strength'] = lgb_pred
# submission


# In[25]:


# submission.to_csv('submission.csv',index=False)
# print('Successfully made prediction')

