#!/usr/bin/env python
# coding: utf-8

# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> 📜 Notebook At a Glance</p>

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> Libraries</p>

# In[65]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)

# Thanks for the idea of CSS @SERGEY SAHAROVSKIY
# Please refer to https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission


# In[64]:


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



from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

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


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> Load dataset</p>

# In[5]:


train = pd.read_csv('/kaggle/input/telecom-churn-case-study-hackathon-c45/train.csv')
test = pd.read_csv('/kaggle/input/telecom-churn-case-study-hackathon-c45/test.csv')


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> Dataset overview and brief EDA</p>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>EDA summary:</u></b><br>
#     
# * <i> There are 172 X variables and 1 target(y) variable, while 1 variable(id) is extra data.</i><br>
# * <i> There are 193,573 rows for train dataset and 30,000 rows for test dataset.</i><br>
# * <i> Some columns have missing values (about 74%).</i><br>
# * <i> The dataset looks like include 3month's user data (Jun, Jul, Aug), also already transformed those to variables.</i><br>
# * <i> For example, arpu6 means Average revenue per user of June while arpu7 is about Average revenue per user of July.</i><br>
# * <i> There are 9 object type variables which indicate date. </i><br>    
#     
# </div>

# In[8]:


train.head()


# In[9]:


train['churn_probability'].value_counts()


# #### > 🧐 let's see the meaning of each variables

# In[10]:


dic = pd.read_csv('/kaggle/input/telecom-churn-case-study-hackathon-c45/data_dictionary.csv')


# In[11]:


dic


# #### > 💡 The dataset looks like include 3month's user data (Jun, Jul, Aug), also already transformed those to variables.
# #### > 💡 In a nutshell, it uses 3month's user behavior data to preidct the probability of churn. (not sure about the definition of churn though)

# In[12]:


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


# #### > ☑️ summary table

# In[13]:


pd.set_option('display.max_rows', None)
summary(train)


# #### > 📊 brief EDA 

# In[14]:


def plot_count(df: pd.core.frame.DataFrame, col_list: list, title_name: str='Train') -> None:

    f, ax = plt.subplots(len(col_list), 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0)
    
    s1 = df[col_list].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#9E3F00', '#eb5e00', '#ff781f', '#ff9752', '#ff9752']
    inner_colors = ['#ff6905', '#ff8838', '#ffa66b']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90,frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={ 'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size':13, 
        'weight': 'bold', 
        'color':'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%',explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', 
                               fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = [0, 1]
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='YlOrBr_r', orient='horizontal'
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
        ax[1].text(v, i+0.1, str(v), color='black', 
                     fontweight='bold', fontsize=12)

#     plt.title(col_list)
    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col_list, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name} Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()    
#     plt.savefig('data/plot_count.png')
    plt.show()


# In[15]:


plot_count(train, ['churn_probability'])


# #### > 💡 the target data is imbalance. but looks not bad from the modeling perspectives. I don't think we have to handle imbalance here.

# In[27]:


# drop object variables which indicates dates in the dataset.
num_cols = test.select_dtypes(include=['float64','int64']).columns.tolist()
num_cols.remove('id')


# In[17]:


len(num_cols)


# In[18]:


# for visualization purpose, let's sample the columns
sample_cols = [col for col in num_cols if '7' in col]


# In[19]:


len(sample_cols)


# In[21]:


get_ipython().run_cell_magic('time', '', "figsize = (18, 32)\nfig = plt.figure(figsize=figsize)\nfor idx, col in enumerate(sample_cols):\n    ax = plt.subplot(17, 3, idx + 1)\n    sns.kdeplot(\n        data=train, hue='churn_probability', fill=True,\n        x=col, palette=['#9E3F00', 'red'], legend=False\n    )\n            \n    ax.set_ylabel(''); ax.spines['top'].set_visible(False), \n    ax.set_xlabel(''); ax.spines['right'].set_visible(False)\n    ax.set_title(f'{col}', loc='right', \n                 weight='bold', fontsize=12)\n\nfig.suptitle(f'Features vs Target\\n\\n\\n', ha='center',  fontweight='bold', fontsize=25)\nfig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=9, ncol=3)\nplt.tight_layout()\nplt.show()\n")


# #### > 💡 As this is imbalanced dataset, it's not easy to check the pattern between churn user and normal user. However, distribution look similar between those two groups. I'm not gonna dig more into this.

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> Simple baseline modeling with LGBM</p>

# In[33]:


# prepare dataset
X = train[num_cols]
Y = train['churn_probability']
FEATURES = num_cols
TARGET = 'churn_probability'


# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Modeliong Overview:</u></b><br>
#     
# * <i> This is super simple version of baseline modeling.</i><br>
# * <i> No variable selection process & No further feature engineering</i><br>
# * <i> Use 5 fold cross validation and use the mean prediction value.</i><br>
# * <i> No hyper-paramter tuning process here.</i><br>
# * <i> In sum, room for improvement : 1) feature selection ; 2) feature engineering ; 3) hyper-parameter tuning ; 4) try different algorithm</i><br>
#     
# </div>

# In[41]:


import time
FOLDS=5
# One of the ground rule for hyper-parameter setting for LGBM is setting small learning_rate with large num_iterations. 
# Also, as there are only 29 X features, so it's better to set max_depth small (4 ~ 8) to avoid over-fitting.
# reference: https://www.kaggle.com/code/odins0n/playground-s-3-e-4-eda-modelling

lgb_params = {
    'objective' : 'binary',
    'metric' : 'auc',
    'learning_rate': 0.002,
    'max_depth': 6,
    'num_iterations': 1000
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
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    
    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    
    X_train, X_valid = train.iloc[train_idx][FEATURES], train.iloc[valid_idx][FEATURES]
    y_train , y_valid = train[TARGET].iloc[train_idx] , train[TARGET].iloc[valid_idx]
    
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
    test_preds = model.predict_proba(test[FEATURES])[:, 1]
    lgb_predictions += test_preds/FOLDS
    
print("Mean AUC :", np.mean(lgb_scores))


# In[46]:


test.head()


# In[54]:


submission = test.copy()
submission['churn_probability'] = np.where(lgb_predictions > 0.5, 1, 0)
submission = submission[['id','churn_probability']]


# In[56]:


submission.head()


# In[66]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




