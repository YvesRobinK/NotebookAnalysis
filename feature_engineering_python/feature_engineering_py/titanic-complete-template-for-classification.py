#!/usr/bin/env python
# coding: utf-8

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> 🛳️ Notebook At a Glance</p>

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
import gc
import re as re
from collections import Counter

from tqdm.auto import tqdm
import math
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

import time
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
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


# load dataset 
# you can copy the path from right menu.
train = pd.read_csv('/kaggle/input/titanic/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('/kaggle/input/titanic/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]

print (train.info())


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Brief EDA</p>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>EDA summary:</u></b><br>
#     
# * <i> There are 11 X variables and 1 target(y) variable, while 1 variable(id) is extra data.</i><br>
# * <i> There are 891 rows for train dataset and 418 rows for test dataset.</i><br>
# * <i> Age, Cabin, Embarked those 3 variables have missing values. -> need to handle missing values</i><br>
# * <i> Combined with categorical and numerical variables. -> need to handle different types of data </i><br>    
#     
# </div>

# #### > 📊 summary table 
# - summary table shows missing value (%), unique value, minimum and maximum value of each variables.
# - Also, you can check each columns' first 3 values, which could help you to get grasp of whole dataset.

# In[4]:


# summary table function
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


summary(test)


# In[7]:


# select numerical and categorical variables respectively.
num_cols = test.select_dtypes(include=['float64','int64']).columns.tolist()
num_cols.remove('PassengerId')
cat_cols = test.select_dtypes(include=['object']).columns.tolist()


# In[8]:


num_cols


# In[9]:


cat_cols


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Exploratory data analysis</p>

# In[10]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);


# In[11]:


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);


# #### > 📊 distribution visualization template for binary classification
# - when it comes to binary classification, it's really important to check distribution of each class
# - comparing distribution bewteen train dataset and test dataset is important too.
# - you can use below two functions to check distribution.

# In[16]:


get_ipython().run_cell_magic('time', '', "figsize = (6*6, 20)\nfig = plt.figure(figsize=figsize)\nfor idx, col in enumerate(num_cols):\n    ax = plt.subplot(2, 3, idx + 1)\n    sns.kdeplot(\n        data=train, hue='Survived', fill=True,\n        x=col, palette=['#9E3F00', 'red'], legend=False\n    )\n            \n    ax.set_ylabel(''); ax.spines['top'].set_visible(False), \n    ax.set_xlabel(''); ax.spines['right'].set_visible(False)\n    ax.set_title(f'{col}', loc='right', \n                 weight='bold', fontsize=20)\n\nfig.suptitle(f'Features vs Target\\n\\n\\n', ha='center',  fontweight='bold', fontsize=25)\nfig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=25, ncol=3)\nplt.tight_layout()\nplt.show()\n")


# In[13]:


import matplotlib.gridspec as gridspec

plt.rc('font', size=12)
grid = gridspec.GridSpec(2,1)
plt.figure(figsize=(6,9))
plt.subplots_adjust(wspace=0.4, hspace=0.3)

features = ['Sex','Embarked']

for idx, feature in enumerate(features):
    ax = plt.subplot(grid[idx])
    
    sns.countplot(x = feature,
                 data = train,
                 hue = 'Survived',
                 palette = 'pastel',
                 ax= ax)
    ax.set_title(f'{feature} Distribution by target')


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Insights:</font></h3>
# 
# * this kind of visualization lead us to undertand the data better. especially when it comes to binary classification, this kind of visualization provide us with many insights. 
# * for example, you can find easily that female passangers survived a lot while most of male passangers died...

# In[17]:


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
    
#     sns.kdeplot(
#         original[column], label='Original',
#         ax=ax[i], color='#20BEFF'
#     )
    
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


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Insights:</font></h3>
# 
# * the distribution bewteen two datasets are similar. but not exactly same.
# * there might be two reasons: 
#     1) dataset is relatively small.
#     2) there might be some outliers!

# #### > 📊 check correlations
# - By viewing correlations among variables, you can understand the relationships between the data.
# - Also, in case you have many variables, you can remove (or select) variables based on corrlation analysis.
# - As this dataset only has small number of X variables, 

# In[18]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    corr = df.corr()
    fig, axes = plt.subplots(figsize=(12, 8))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrRd', annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(train, 'Train Dataset Correlation')
plot_correlation_heatmap(test, 'Test Dataset Correlation')


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Feature engineering</p>

# #### > ❗ outlier detection with tukey method
# 
# - Based on our EDA, we found that there might be outliers on train set.
# - We use tukey method to handle outliers in this notebook.

# ![image.png](attachment:c0016fa8-d7a5-4bdb-812e-08837846137d.png)

# In[19]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

# kodus to https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling


# In[20]:


# drop outliers!
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# #### > ⛏️ transform data - encoding
# - The LabelEncoder in Scikit-learn will convert each unique string value into a number, making out data more flexible for various algorithms.
# 
# ![image.png](attachment:677cd28f-05ea-493f-9205-40b2ff67f447.png)

# In[21]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(train)
data_test = transform_features(test)
data_train.head()

# kudos to https://www.kaggle.com/code/jeffd23/scikit-learn-ml-from-start-to-finish


# In[22]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
data_train.head()


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Modeling</p>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Modeling overview:</u></b><br>
#     
# * <i> build baseline model without hyperparameter tuning</i><br>
# * <i>  5-fold cross validation methods are used for baseline modeling.</i><br>
# * <i> * Evalution metric is AUC</i><br>
#   
# </div>

# ![image.png](attachment:cb3d58e4-95d8-44e9-8209-f9a08507dab6.png)

# In[23]:


X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# In[24]:


TARGET = 'Survived'
FEATURES = [col for col in X_all.columns if col != TARGET]


# In[25]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
# One of the ground rule for hyper-parameter setting for LGBM is setting small learning_rate with large num_iterations. 
# Also, as there are only 18 X features, so it's better to set max_depth small (4 ~ 6) to avoid over-fitting.

# set CV fold
FOLDS=5

# set hyper-parameter 
lgb_params = {
    'objective' : 'binary',
    'metric' : 'auc',
    'learning_rate': 0.002,
    'max_depth': 6,
    'num_iterations': 1500,
    'feature_fraction': 0.8247273276668773,
    'bagging_fraction': 0.5842711778104962
}



# lgb_params ={'objective': 'binary',
#              'metric': 'auc',
#              'lambda_l1': 1.0050418664783436e-08, 
#              'lambda_l2': 9.938606206413121,
#              'num_leaves': 44,
#              'feature_fraction': 0.8247273276668773,
#              'bagging_fraction': 0.5842711778104962,
#              'bagging_freq': 6,
#              'min_child_samples': 70,
#              'max_depth': 8,
#              'num_iterations': 400,
#              'learning_rate':0.05}

lgb_predictions = 0
lgb_scores = []
lgb_imp = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1004)
for fold, (train_idx, valid_idx) in enumerate(skf.split(data_train[FEATURES], data_train[TARGET])):
    
    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    
    X_train, X_valid = data_train.iloc[train_idx][FEATURES], data_train.iloc[valid_idx][FEATURES]
    y_train , y_valid = data_train[TARGET].iloc[train_idx] , data_train[TARGET].iloc[valid_idx]
    
    model = LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train,verbose=0)
    
    preds_valid = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid,  preds_valid)
    lgb_scores.append(auc)
    run_time = time.time() - start_time
    
    print(f"Fold={fold+1}, AUC score: {auc:.2f}, Run Time: {run_time:.2f}s")
    fim = pd.DataFrame(index=FEATURES,
                 data=model.feature_importances_,
                 columns=[f'{fold}_importance'])
    lgb_imp.append(fim)
    test_preds = model.predict_proba(X_test[FEATURES])[:, 1]
    lgb_predictions += test_preds/FOLDS
    
print("Mean AUC :", np.mean(lgb_scores))


# In[27]:


fim.plot(kind = 'barh');
plt.xlabel('Feature importance score')
plt.ylabel('Features')
plt.show(); 


# #### > 🎰 let's check model performance on test dataset

# In[28]:


roc_auc_score(y_test, lgb_predictions)


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Conclusion:</font></h3>
# 
# * mean AUC on train set : 86.8%
# * performance on test set: 93.14%

# #### > ✅ references:
# 
# https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling
#     
# https://www.kaggle.com/code/jeffd23/scikit-learn-ml-from-start-to-finish
#     
# https://www.kaggle.com/code/jeffd23/scikit-learn-ml-from-start-to-finish
