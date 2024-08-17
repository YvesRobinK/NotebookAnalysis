#!/usr/bin/env python
# coding: utf-8

# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> 📜 Notebook At a Glance</p>

# ![image.png](attachment:b4a2bfa6-1330-4c53-9241-ff1653465c4f.png)

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)

# Thanks for the idea of CSS @SERGEY SAHAROVSKIY
# Please refer to https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from tqdm.auto import tqdm
import math
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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


# In[3]:


train = pd.read_csv("../input/playground-series-s3e8/train.csv").drop(columns='id')
test = pd.read_csv("../input/playground-series-s3e8/test.csv").drop(columns='id')
origin = pd.read_csv("../input/gemstone-price-prediction/cubic_zirconia.csv", index_col = 0)


# 
# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Brief EDA</p>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>EDA summary:</u></b><br>
#     
# * <i> There are 10 X variables and 1 target(y) variable, while 1 variable(id) is extra data.</i><br>
# * <i> There are 193,573 rows for train dataset and 129,050 rows for test dataset.</i><br>
# * <i> No missing values.</i><br>
# * <i> 3 variables (cut, color, clarity) are object type while others are float64 type. </i><br>    
#     
# </div>

# In[4]:


def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


# In[5]:


summary(train)


# In[6]:


# select numerical and categorical variables respectively.
num_cols = test.select_dtypes(include=['float64','int64']).columns.tolist()
cat_cols = test.select_dtypes(include=['object']).columns.tolist()
all_cols = num_cols + cat_cols


# > ##### 📊 distribution of target value

# In[7]:


sns.displot(train, x="price")


# In[8]:


sns.displot(origin, x="price")


# > ##### 📊 EDA of other numerical variables 

# In[9]:


# kudos to @jcaliz /  
# refer to https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission
features = num_cols
n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}

columns = features
n_cols = 4
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    sns.kdeplot(
        train[column], label='Train',
        ax=ax[i], color='#9E3F00'
    )
    
    sns.kdeplot(
        test[column], label='Test',
        ax=ax[i], color='yellow'
    )
    
    sns.kdeplot(
        origin[column], label='Original',
        ax=ax[i], color='#20BEFF'
    )
    
    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    
    # remove axes to show only one at the end
    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]
        plot_ax.legend().remove()
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')
    
fig.suptitle(f'Numerical Feature Distributions\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=25, ncol=3)
plt.tight_layout()


# In[10]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    corr = df.corr()
    fig, axes = plt.subplots(figsize=(20, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrRd', annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(origin, 'Original Dataset Correlation')
plot_correlation_heatmap(train, 'Train Dataset Correlation')
plot_correlation_heatmap(train, 'Test Dataset Correlation')


# 
# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Feature Egnineering</p>

# > ##### conduct minimal data engineering job for XGB baseline

# In[11]:


origin.head()


# In[12]:


train.head()


# In[13]:


df = pd.concat([train, origin])


# In[14]:


print( f'data shape of original dataset:' ,origin.shape)
print( f'data shape of train dataset:' ,train.shape)
print( f'data shape after merging original dataset:' ,df.shape)


# In[15]:


# create dummies with categorical variables
dummies = pd.get_dummies(df[['cut', 'color', 'clarity']])
X = df.drop(columns = ['cut', 'color', 'clarity', 'price'], axis = 1)
X = pd.concat([X, dummies], axis = 1)
Y = df['price']


# In[16]:


dummies_test = pd.get_dummies(test[['cut', 'color', 'clarity']])
test = pd.concat([test.drop(columns = ['cut', 'color', 'clarity'], axis = 1), dummies_test], axis = 1)


# 
# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Baseline modeling with XGB</p>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>modeling overview:</u></b><br>
#     
# * <i> build baseline model without hyperparameter tuning.</i><br>
# * <i> 3-fold cross validation methods are used for baseline modeling.</i><br>
# * <i> Evalution metric is Root Mean Squared Error</i><br>
#     
# </div>

# ![image.png](attachment:d58543c6-b55a-4aeb-8cea-5647289f76d1.png)

# In[17]:


#credit : https://www.kaggle.com/code/oscarm524/ps-s3-ep8-eda-modeling
from sklearn.metrics import mean_squared_error

cv_scores = list()
importance_xgb = list()
preds = list()

## Running 3 fold CV
for i in range(3):
    print(f'{i} fold cv begin')
    skf = KFold(n_splits = 3, random_state = 1004, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBRegressor(tree_method = 'gpu_hist',
                              colsample_bytree = 0.8, 
                              gamma = 0.8, 
                              learning_rate = 0.01, 
                              max_depth = 6, 
                              min_child_weight = 10, 
                              n_estimators = 1000, 
                              subsample = 0.8).fit(X_train, Y_train)
        importance_xgb.append(XGB_md.feature_importances_)
        
        XGB_pred_1 = XGB_md.predict(X_test)
        XGB_pred_2 = XGB_md.predict(test)
        
        # Calculate RMSE
        cv_scores.append(mean_squared_error(Y_test, XGB_pred_1, squared = False))
        preds.append(XGB_pred_2)
        print(f'{i} fold cv done')

scores = np.mean(cv_scores)    
print('The average RMSE over 3-folds (run 3 times) is:', scores)


# > ##### The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. 

# ![image.png](attachment:f885deec-025b-43be-9126-653023247483.png)

# In[18]:


plt.figure(figsize = (8, 8))
pd.DataFrame(importance_xgb, columns = X.columns).apply(np.mean, axis = 0).sort_values().plot(kind = 'barh');
plt.xlabel('Feature importance score')
plt.ylabel('Features')
plt.show(); 


# 
# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Submission</p>

# > ##### submit mean value of prediction
# > ##### I recommend that you re-train the model and select best parameter. 

# In[19]:


preds_mean_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
test_submit = pd.read_csv("../input/playground-series-s3e8/test.csv", usecols = ['id'])
test_submit['price'] = preds_mean_test
test_submit.head()


# In[20]:


test_submit.to_csv('submission.csv', index = False)


# In[21]:


# to be continued....

