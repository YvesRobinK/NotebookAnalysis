#!/usr/bin/env python
# coding: utf-8

# 
# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000"> Introduction</p>

# ### Welcome to the Kaggle Playground Series Season 3 Episode 23!

# ### Information About Competition:

# The challenge in this competition revolves around binary classification for detecting software defects using synthetic dataset that was generated from a deep learning model trained on the [Software Defect Dataset](https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction?select=about+JM1+Dataset.txt). The performance metric employed to assess the models is the [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (Receiver Operating Characteristic curve).

# ### Information About Dataset:

# The Dataset contains of 22 features that are based on 2 code measures: 
# 
# * McCabe Code Measures:
# 
#     McCabe's code measures, also known as cyclomatic complexity, help us understand how complex a piece of code is. Imagine your code as a flowchart, with different paths and decision points. Cyclomatic complexity counts how many different paths there are through this flowchart. A higher cyclomatic complexity indicates that the code has more decision points and is more intricate or harder to understand. It's like counting the number of different ways you can navigate through a maze - a simple maze has fewer routes, while a complex one has many.
# 
# * Halstead Code Measures:
# 
#     Halstead code measures focus on the size and complexity of the code in terms of operators and operands. Operators are things like +, -, *, /, while operands are variables and constants. Halstead's measures include metrics like the number of distinct operators and operands, the total number of operator occurrences, and the total number of operand occurrences. By looking at these metrics, you can get a sense of how "big" and complex a piece of code is based on the number of different operations and variables used. Think of it as counting the number of different tools (operators) and materials (operands) used in building something - more variety usually means more complexity.
#     
# In essence, both McCabe and Halstead measures provide insights into different aspects of code complexity: McCabe into how complicated the code's logical structure is, and Halstead into how much stuff (operators and operands) is in the code. 
# 

# ### More About McCabe Code Measures:

# In[1]:


from IPython.display import HTML

HTML('''<div align="center">
<iframe align = "middle"width="790"
height="440"
src="https://www.youtube.com/embed/PDYmEtBSn60"
title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe></div>'
     ''')


# ### More About Halstead Code Measures:

# In[2]:


HTML('''<div align="center">
<iframe align = "middle"width="790"
height="440"
src="https://www.youtube.com/embed/2bjMmjOwSuE"
title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe></div>'
     ''')


# ### Data Describtion:

# Here is a describtion of data taken from `about JM1 Dataset.txt` document from the original data source on Kaggle (https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction?select=about+JM1+Dataset.txt).
# 
# | Column Name | Description |
# | ----------- | ----------- |
# | loc              | McCabe's line count of code      
# | v(g)             | McCabe "cyclomatic complexity"      
# | ev(g)            | McCabe "essential complexity"       
# | iv(g)            | McCabe "design complexity"          
# | n                | Halstead total operators + operands 
# | v                | Halstead "volume"
# | l                | Halstead "program length"
# | d                | Halstead "difficulty"
# | i                | Halstead "intelligence"
# | e                | Halstead "effort"
# | b                | Halstead
# | t                | Halstead's time estimator
# | lOCode           | Halstead's line count
# | lOComment        | Halstead's count of lines of comments
# | lOBlank          | Halstead's count of blank lines
# | lOCodeAndComment | numeric
# | uniq_Op          | numeric unique operators
# | uniq_Opnd        | numeric  unique operands
# | total_Op         | numeric  total operators
# | total_Opnd       | numeric  total operands
# | branchCount      | numeric  of the flow graph
# | defects          | {false,true}  module has/has not one or more reported defects 
# 

# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000"> Import</p>

# In[3]:


# Misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
import gc
import warnings
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool


# In[4]:


# Seaborn
rc = {
    #FAEEE9
    "axes.facecolor": "#243139",
    "figure.facecolor": "#243139",
    "axes.edgecolor": "#000000",
    "grid.color": "#000000",
    "font.family": "arial",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.alpha": 0.4
}
sns.set(rc=rc)

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Functions
def print_sl():
    print("=" * 50)
    print()


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000"> Load Data</p>

# In[5]:


train = pd.read_csv('/kaggle/input/playground-series-s3e23/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e23/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')

train_orig = pd.read_csv('/kaggle/input/software-defect-prediction/jm1.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

print('Data Loaded Succesfully!')
print_sl()

print(f'train shape: {train.shape}')
print(f'are there any null values in train: {train.isnull().any().any()}\n')

print(f'test shape: {test.shape}')
print(f'are there any null values in test: {test.isnull().any().any()}\n')

print(f'train_orig shape: {train_orig.shape}')
print(f'are there any null values in test: {train_orig.isnull().any().any()}\n')

num_cols = train.columns.delete(-1)

target = 'defects'

train.head()


# 📌 **Note**:
# * Both the original and synthesized datasets are complete, with no missing values.

# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">EDA</p>

# In[6]:


# https://www.kaggle.com/code/kimtaehun/eda-and-baseline-with-multiple-models
def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    # Set background color
    
    f, ax = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#59b3a3', '#433C64']
    inner_colors = ['#59b3a3', '#433C64']
    #['#ff6905', '#ff8838', '#ffa66b']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='#243139', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='mako', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,       
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='white', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='white')
    ax[1].set_ylabel('count', fontweight="bold", color='white')

    f.suptitle(f'{title_name}', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout() 
    plt.show()


# ### Target Variable Distribution of Synthesized Data:

# In[7]:


plot_count(train, 'defects', 'Target Variable Distribution of Synthesized Data')


# ### Target Variable Distribution of Original Data:

# In[8]:


plot_count(train_orig, 'defects', 'Target Variable Distribution of Original Data')


# 📌 **Note**:
# * Clear class disbalance in both the original and synthesized datasets.

# ### Distribution of Variables in Synthesized Dataset:

# In[9]:


plt.figure(figsize=(16, len(num_cols) * 3))

for i, col in enumerate(num_cols):
    # Plotting for outcome
    plt.subplot(len(num_cols), 2, i+1)
    sns.histplot(x=col, hue="defects", data=train, bins=30, kde=True, palette='mako')
    plt.title(f"{col} distribution for outcome", fontweight="bold", color = 'white')
    plt.ylim(0, train[col].value_counts().max() + 10)
    
    
    plt.legend(title="Defects", loc='upper right', labels=['true', 'false'], labelcolor='white').get_title().set_color("white")
    
plt.tight_layout()
plt.show()


# ### Distribution of Variables in Original Dataset:

# In[10]:


plt.figure(figsize=(16, len(num_cols) * 3))

for i, col in enumerate(num_cols):
    # Plotting for outcome
    plt.subplot(len(num_cols), 2, i+1)
    sns.histplot(x=col, hue="defects", data=train_orig, bins=30, kde=True, palette='mako')
    plt.title(f"{col} distribution for outcome", fontweight="bold", color = 'white')
    plt.ylim(0, train_orig[col].value_counts().max() + 10)
    
    
    plt.legend(title="Defects", loc='upper right', labels=['true', 'false'], labelcolor='white').get_title().set_color("white")
    
plt.tight_layout()
plt.show()


# In[11]:


my_palette = 'mako'
# Create subplots
fig, axes = plt.subplots(len(num_cols), 2, figsize=(16, len(num_cols) * 3))

# Plot the histograms and box plots
for i, column in enumerate(num_cols):
    # Histogram
    sns.histplot(train[column], bins=30, kde=True, ax=axes[i, 0], palette = my_palette)
    axes[i, 0].set_title(f'Distribution of {column} in train', color='white')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    # Box plot
    sns.boxplot(train[column], ax=axes[i, 1], palette = my_palette)
    axes[i, 1].set_title(f'Box plot of {column} in train', color='white')
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()


# 📌 **Note**:
# * We can see that for both the original and synthesized datasets data is right-skewed with lots of outliers.

# In[12]:


# Create a copy of the dataframe
df = train.copy()

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='mako', annot=True, annot_kws={"size": 6})
    plt.title(title_name, color='white')
    plt.show()

# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df, 'Dataset Correlation')


# 📌 **Note**:
# * Lots of features strongly correlate with each other, however there is no strong correlation between target and saperate features.

# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">Feature Engineering</p>

# In[13]:


total = pd.concat([train, train_orig], ignore_index=True)
total.drop_duplicates(inplace=True)

total.dtypes


# In[14]:


for column in ['uniq_Op', 'uniq_Opnd', 'total_Op','total_Opnd', 'branchCount']:
    print(len(total[total[column] == '?']), total[total[column] == '?'].index)


# 📌 **Note**:
# * We can see that uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount are object dtypes. The reason is that some of the values are '?'.
# * There are only 5 rows with '?' therefore I decided to drop them.

# In[15]:


for column in ['uniq_Op', 'uniq_Opnd', 'total_Op','total_Opnd', 'branchCount']:
    total.drop(total[total[column] == '?'].index, inplace=True)
    total[column] = total[column].astype(float)
    
    test.drop(test[test[column] == '?'].index, inplace=True)
    test[column] = test[column].astype(float)


# In[16]:


test.reset_index()


# In[17]:


X_train = total.drop(columns=[target]).reset_index().drop(columns=['index'])
y_train = total.defects.astype(int).reset_index().drop(columns=['index'])
X_test = test.reset_index().drop(columns=['index'])

# for column in X_train.columns:
#     X_train[column] = X_train[column].apply(lambda x: np.log(x) if x > 0 else 0)
#     X_test[column] = X_test[column].apply(lambda x: np.log(x) if x > 0 else 0)

def scale(x):
    scaler = preprocessing.RobustScaler()
    robust_df = scaler.fit_transform(x)
    robust_df = pd.DataFrame(robust_df, columns =x.columns)
    return robust_df
X_train = scale(X_train)
X_test = scale(X_test)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')

X_train.head()


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">Model Building</p>

# In[18]:


# Weights for class disbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

class_weight_0 = 1.0
class_weight_1 = 1.0 / scale_pos_weight

class_weights_cat = [class_weight_0, class_weight_1]

class_weights_lgb = {0: class_weight_0, 1: class_weight_1}


# In[19]:


class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold == 'skf':
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    if type(X) is np.ndarray:
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                    else:
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            raise ValueError(f"Invalid kfold: Must be True")


# In[20]:


class Classifier:
    def __init__(self, n_estimators=200, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_optuna0 = {
            'n_estimators': 1000,
            'learning_rate': 0.01752354328845971,
            'booster': 'gbtree',
            'lambda': 0.08159630121074074,
            'alpha': 0.07564858712175693,
            'subsample': 0.5065979400270813,
            'colsample_bytree': 0.6187340851873067,
            'max_depth': 4,
            'min_child_weight': 5,
            'eta': 0.2603059902806757,
            'gamma': 0.6567360773618207,
            #'scale_pos_weight': scale_pos_weight,
            'random_state': random_state
        }
        
        lgb_params0 = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'goss',
            'random_state': 42,
            'colsample_bytree': 0.50,
            'subsample': 0.70,
            'learning_rate': 0.0625,
            'max_depth': 6,
            'n_estimators': 1000,
            'num_leaves': 110, 
            'reg_alpha': 0.0001,
            'reg_lambda': 2.0,
            'verbose': -1,
            'random_state': random_state,
        }


        
    ### All those models are from previous binary classification competitions that I participated in. They are not tuned for this particular competition and I use them for baseline solution    
        xgb_params0 = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.09641232707445854,
            'booster': 'gbtree',
            'lambda': 4.666002223704784,
            'alpha': 3.708175990751336,
            'subsample': 0.6100174145229473,
            'colsample_bytree': 0.5506821152321051,
            'max_depth': 7,
            'min_child_weight': 3,
            'eta': 1.740374368661041,
            'gamma': 0.007427363662926455,
            'grow_policy': 'depthwise',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': self.random_state,
            #'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params1 = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.012208383405206188,
            'booster': 'gbtree',
            'lambda': 0.009968756668882757,
            'alpha': 0.02666266827121168,
            'subsample': 0.7097814108897231,
            'colsample_bytree': 0.7946945784285216,
            'max_depth': 3,
            'min_child_weight': 4,
            'eta': 0.5480204506554545,
            'gamma': 0.8788654128774149,
            'scale_pos_weight': 4.71,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }

        
        
        xgb_params2 = {
            'n_estimators': self.n_estimators,
            'colsample_bytree': 0.5646751146007976,
            'gamma': 7.788727238356553e-06,
            'learning_rate': 0.1419865761603358,
            'max_bin': 824,
            'min_child_weight': 1,
            'random_state': 811996,
            'reg_alpha': 1.6259583347890365e-07,
            'reg_lambda': 2.110691851528507e-08,
            'subsample': 0.879020578464637,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'n_jobs': -1,
            'verbosity': 0,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params3 = {
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'colsample_bytree': 0.4836462317215041,
            'eta': 0.05976752607337169,
            'gamma': 1,
            'lambda': 0.2976432557733288,
            'max_depth': 6,
            'min_child_weight': 1,
            'n_estimators': 550,
            'objective': 'binary:logistic',
            'scale_pos_weight': 4.260162886376033,
            'subsample': 0.7119282378433924,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params4 = {
            'n_estimators': self.n_estimators,
            'colsample_bytree': 0.8757972257439255,
            'gamma': 0.11135738771999848,
            'max_depth': 7,
            'min_child_weight': 3,
            'reg_alpha': 0.4833998914998038,
            'reg_lambda': 0.006223568555619563,
            'scale_pos_weight': 8,
            'subsample': 0.7056434340275685,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        xgb_params5 = {
            'n_estimators': self.n_estimators,
            'max_depth': 5, 
            'min_child_weight': 2.934487833919741,
            'learning_rate': 0.11341944575807082, 
            'subsample': 0.9045063514419968,
            'gamma': 0.4329153382843715,
            'colsample_bytree': 0.38872702868412506,
            'colsample_bylevel': 0.8321880031718571,
            'colsample_bynode': 0.802355707802605,
            'random_state': self.random_state,
            #'scale_pos_weight': scale_pos_weight
       }
        
        xgb_base = {
            'n_estimators': self.n_estimators,
           # 'scale_pos_weight': scale_pos_weight,
            'verbosity': 0,
            'random_state': self.random_state,
        }
        
        xgb_params6 = {
            'objective': 'binary:logistic',
            'colsample_bytree': 0.7, 
            'gamma': 2, 
            'learning_rate': 0.01, 
            'max_depth': 7, 
            'min_child_weight': 10, 
            'n_estimators': 500, 
            'subsample':0.7,
            'random_state': self.random_state,
           # 'scale_pos_weight': scale_pos_weight
        }
        
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
       
        models = {
            
            # XGBoost
            'xgbOp': xgb.XGBClassifier(**xgb_optuna0),
           # 'xgb0': xgb.XGBClassifier(**xgb_params0),
            'xgb1': xgb.XGBClassifier(**xgb_params1),
            #'xgb2': xgb.XGBClassifier(**xgb_params2),
            'xgb3': xgb.XGBClassifier(**xgb_params3),
           # 'xgb4': xgb.XGBClassifier(**xgb_params4),
            'xgb5': xgb.XGBClassifier(**xgb_params5),
            'xgbb': xgb.XGBClassifier(**xgb_base),
            'xgb6': xgb.XGBClassifier(**xgb_params6),
            
            # Misc
            'lgb0': lgb.LGBMClassifier(**lgb_params0),
            
            # add some models with default params to "simplify" ensemble
           # 'svc': SVC(random_state=self.random_state, probability=True),
           # 'brf': BalancedRandomForestClassifier(max_depth = 5, random_state=self.random_state),

        }
        
        return models


# In[21]:


class OptunaWeights:
    def __init__(self, random_state, n_trials=500):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-14, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        # score = log_loss(y_true, weighted_pred)
        score = roc_auc_score(y_true, weighted_pred)
        
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# ### Training:

# In[22]:


get_ipython().run_cell_magic('time', '', "\n# Config\nkfold = 'skf'\nn_splits = 5\nn_reapts = 3\nrandom_state = 42\nn_estimators = 999\nearly_stopping_rounds = 999\nverbose = False\ndevice = 'cpu'\n\n# Fix seed\nrandom.seed(random_state)\nrandom_state_list = random.sample(range(9999), n_reapts)\n#random_state_list = [42]\n\n# Initialize an array for storing test predictions\nclassifier = Classifier(n_estimators, device, random_state)\ntest_predss = np.zeros((X_test.shape[0]))\noof_predss = np.zeros((X_train.shape[0], n_reapts))\n\n# Store scores and weights\nensemble_score = []\nweights = []\n\n# Predictions and models\noof_each_predss = []\noof_each_preds = np.zeros((X_train.shape[0], classifier.len_models))\ntest_each_predss = []\ntest_each_preds = np.zeros((X_test.shape[0], classifier.len_models))\ntrained_models = {'xgb':[],}\nscore_dict = dict(zip(classifier.models_name, [[] for _ in range(classifier.len_models)]))\n\n# Loop over KFold splits\nsplitter = Splitter(kfold=kfold, n_splits=n_splits)\nfor i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):\n    n = i % n_splits\n    m = i // n_splits\n            \n    # Get a set of classifier models\n    classifier = Classifier(n_estimators, device, random_state_list[m])\n    models = classifier.models\n    \n    # Initialize lists to store oof and test predictions for each base model\n    oof_preds = []\n    test_preds = []\n    \n    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions\n    for name, model in models.items():\n        if ('xgb' in name) or ('lgb' in name) or ('cat' in name):\n            if 'xgb' in name:\n                model.fit(\n                    X_train_, y_train_, \n                    eval_set=[(X_val, y_val)],\n                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)\n            elif 'lgb' in name:\n                model.fit(\n                    X_train_, y_train_, \n                    eval_set=[(X_val, y_val)],\n                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)\n            elif 'cat' in name:\n                model.fit(\n                    Pool(X_train_, y_train_), \n                    eval_set=Pool(X_val, y_val),\n                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)\n        else:\n            model.fit(X_train_, y_train_)\n            \n        if name in trained_models.keys():\n            trained_models[f'{name}'].append(deepcopy(model))\n        \n        test_pred = model.predict_proba(X_test)[:, 1]\n        y_val_pred = model.predict_proba(X_val)[:, 1]\n        \n        # Calculate recall and precision scores\n        y_val_pred_binary = (y_val_pred > 0.5).astype(int)\n        recall = recall_score(y_val, y_val_pred_binary)\n        precision = precision_score(y_val, y_val_pred_binary)\n        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Recall score: {recall:.5f}')\n        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Precision score: {precision:.5f}')\n\n        score = roc_auc_score(y_val, y_val_pred)\n        score_dict[name].append(score)\n        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC score: {score:.5f}')\n        print('-'*50)\n        \n        oof_preds.append(y_val_pred)\n        test_preds.append(test_pred)\n    \n    # Use Optuna to find the best ensemble weights\n    optweights = OptunaWeights(random_state=random_state_list[m])\n    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)\n    \n    score_ = roc_auc_score(y_val, y_val_pred)\n    print(f'--> Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ROC score {score_:.5f}')\n    print_sl()\n    ensemble_score.append(score_)\n    weights.append(optweights.weights)\n    \n    # Predict to X_test by the best ensemble weights\n    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))\n    oof_predss[X_val.index, m] += optweights.predict(oof_preds)\n    oof_each_preds[X_val.index] = np.stack(oof_preds).T\n    test_each_preds += np.array(test_preds).T / n_splits\n    \n    if n == (n_splits - 1):\n        oof_each_predss.append(oof_each_preds)\n        oof_each_preds = np.zeros((X_train.shape[0], classifier.len_models))\n        test_each_predss.append(test_each_preds)\n        test_each_preds = np.zeros((X_test.shape[0], classifier.len_models))\n    \n    gc.collect()\n    \noof_each_predss = np.mean(np.array(oof_each_predss), axis=0)\ntest_each_predss = np.mean(np.array(test_each_predss), axis=0)\noof_each_predss = np.concatenate([oof_each_predss, np.mean(oof_predss, axis=1).reshape(-1, 1)], axis=1)\ntest_each_predss = np.concatenate([test_each_predss, test_predss.reshape(-1, 1)], axis=1)\n")


# In[23]:


# Calculate the mean score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Mean Optuna Ensemble {mean_score:.5f} ± {std_score:.5f} \n')

print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# ### Stacking Preds:

# In[24]:


get_ipython().run_cell_magic('time', '', "\nstack_test_predss = np.zeros((X_test.shape[0]))\nstack_scores = []\nstack_models = []\nsplitter = Splitter(kfold=kfold, n_splits=n_splits)\nfor i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(oof_each_predss, np.array(y_train), random_state_list=random_state_list)):\n    n = i % n_splits\n    m = i // n_splits\n    \n    classifier = Classifier(n_estimators, device, random_state_list[m])\n    models = classifier.models\n    model = models['xgb3']\n    \n    model.fit(\n    X_train_, y_train_,\n    eval_set=[(X_val, y_val)],\n    early_stopping_rounds=early_stopping_rounds,\n    verbose=verbose\n)\n    \n    test_pred = model.predict_proba(test_each_predss)[:, 1]\n    y_val_pred = model.predict_proba(X_val)[:, 1]\n\n    score = roc_auc_score(y_val, y_val_pred)\n    stack_scores.append(score)\n    stack_models.append(deepcopy(model))\n    \n    stack_test_predss += test_pred / (n_splits * len(random_state_list))\n")


# In[25]:


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble ROC score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')
print('')

# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(stack_scores)
std_score = np.std(stack_scores)
print(f'Stacking ROC score {mean_score:.5f} ± {std_score:.5f}\n')


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">Submission</p>

# In[26]:


sub = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')

sub['defects'] = stack_test_predss
sub.to_csv('submission.csv', index=False)
sub

