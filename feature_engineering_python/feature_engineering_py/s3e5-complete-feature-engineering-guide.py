#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction
# 
# > This is a companion to my previous work [HERE](https://www.kaggle.com/code/shilongzhuang/s3e5-training-with-data-from-uci), which is a beginner-friendly walkthough of my solution.
# 
# ![](https://media.tenor.com/NF6ixwAmrTMAAAAd/cristiano-ronaldo-drinking.gif)

# # Import Libraries

# In[1]:


# data analysis
import pandas as pd
import numpy as np
import math

# data visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# modelling
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import time
import optuna
import xgboost as xgb
import lightgbm as lgbm
import catboost

import statsmodels.api as sm


# # Acquire Data

# In[2]:


# Original dataset
orig = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv')

# Additional datasets from UCI repository
redwine = pd.read_csv('/kaggle/input/red-and-white-wine-quality/winequality-red.csv')
whitewine = pd.read_csv('/kaggle/input/red-and-white-wine-quality/winequality-white.csv')

# datasets provided from competition
train = pd.read_csv('/kaggle/input/playground-series-s3e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s3e5/sample_submission.csv')

orig = orig[[c for c in orig.columns if c not in ['Id']]]
train = train[[c for c in train.columns if c not in ['Id']]]
test = test[[c for c in test.columns if c not in ['Id']]]

train = pd.concat([train, orig, redwine, ]).reset_index(drop=True)
train.drop_duplicates(inplace=True)


# # Set Theme

# In[3]:


wine2 = ['#F9DBBD', '#FFA5AB', '#DA627D', '#A53860', '#450920', ]
colors = wine2

sns.set_theme(
    style='darkgrid',
)


# In[4]:


train.rename(lambda x: x.lower().strip().replace(' ', '_'), axis='columns', inplace=True)
test.rename(lambda x: x.lower().strip().replace(' ', '_'), axis='columns', inplace=True)


# In[5]:


orig.head()


# In[6]:


redwine.head()


# In[7]:


whitewine.head()


# In[8]:


train.head()


# # Correlation
# 
# Seting up the correlation matrix allows to uncover dependencies across different features and the target, which can be helpful for providing insights in engineering our features.

# In[9]:


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

def corr_matrix(dataframe, x, y):
    plt.subplots(figsize=(x, y))
    return sns.heatmap(dataframe.corr(), cmap=cmap,
            annot=True,
            linewidths=3,
            annot_kws={"fontsize":13},
            square=True,
           )

corr_matrix(train, 15, 15)


# # Feature Engineering
# 
# Feature engineering is the process of creating new features or modifying existing features in your data to improve the performance of a machine learning model. The correlation matrix wouldn't be enough to give us all we need. Researching articles can help in this process by providing insights and ideas on how to transform and create new features that may have a significant impact on the model's prediction.

# ## Alcohol * Density
# 
# Following the Density formula, the alcohol content in terms of mass can be computed by multiplying `alcohol`, in the form of percentage by volume, with `density`. This will be treated as a new feature called `alcohol_density`.
# 
# ![image.png](attachment:c397e8c8-a112-425b-a4c1-724533934fd6.png)!

# In[10]:


train['alcohol_density'] = train.alcohol * train.density


# ## Total Acidity
# 
# Acidity is a characteristic determined by the total sum of acids that a sample contains. It is one of the important elements to monitor during winemaking because they give us an indication of what is going on with the overall balance of the wine. This can be quantified by taking the sum of all the acids that correspond to `fixed_acidity`, `volatile_acidity`, and `citric_acid`.
# - `fixed_acidity` corresponds to the set of low volatility organic acids such as malic, lactic, tartaric or citric acids and is inherent to the characteristics of the sample
# - `volatile_acidity` corresponds to the set of short chain organic acids that can be extracted from the sample by means of a distillation process: formic acid, acetic acid, propionic acid and butyric acid.

# In[11]:


train['total_acidity'] = train.volatile_acidity + train.fixed_acidity + train.citric_acid

# experimenting with this feature
train['percent_acidity'] = (train.volatile_acidity / (train.fixed_acidity) + train.citric_acid)


# ## Ideal ph Level
# 
# According to this [source](https://morewinemaking.com/articles/testing_wine_must), the ideal ph level for red wine should be around the 3.4 to 3.6 range. A `pH` above 3.6 indicates and unstable wine and will not have a long shelf life. pH under 3.4 generally indicates a wine that will be too sour. This may serve as a relevant indicator of what differentiated a good from a bad red wine.

# In[12]:


train['ideal_ph'] = 0
train.loc[(train.ph >= 3.4) & (train.ph <= 3.65), 'ideal_ph'] = 1


# ## Acidity * ph

# In[13]:


train['ph_acidity'] = train.ph * train.percent_acidity


# ## Percent Free SO2
# 
# `sulfur_dioxide` is used in winemaking as a preservative to prevent oxidation and microbial spoilage. It exists in three forms; bisulfite (HSO3-), molecular SO2, and sulfite (SO32). The equilibrium is pH dependent with the predominate form at wine pH being bisulfite. Most of the rest is molecular and very little, if any, remains in sulfite form. These forms make up what is termed as `free_sulfur_dioxide`. `free_sulfur_dioxide` can be lost through volatilization or binding, thus management is important.
# 
# ![image.png](attachment:3dc56151-460e-4d4f-850b-a779a62b1672.png)

# In[14]:


train['percent_free_sulfur'] = train.free_sulfur_dioxide / train.total_sulfur_dioxide


# In[15]:


train.columns


# In[16]:


test['percent_acidity'] = (test.volatile_acidity / (test.fixed_acidity) + test.citric_acid)
test['percent_free_sulfur'] = test.free_sulfur_dioxide / test.total_sulfur_dioxide
test['alcohol_density'] = test.alcohol * test.density


# ## Correlation after Feature Engineering

# In[17]:


corr_matrix(train, 18, 18)


# ## Remove Multicollinear Features

# In[18]:


cols = ['fixed_acidity',
        'volatile_acidity',
        'citric_acid',
        'residual_sugar',
       'chlorides',
        'free_sulfur_dioxide',
        'total_sulfur_dioxide',
        'density',
       'ph',
        'sulphates',
        'alcohol',
        'quality',
        'total_acidity',
       'alcohol_density',
        'ph_acidity',
        'ideal_ph',
       'percent_free_sulfur',
        'percent_acidity']

_ = train[[
#     'fixed_acidity',
#     'volatile_acidity',
#     'citric_acid',  
#     'residual_sugar', 
    'chlorides',
#     'free_sulfur_dioxide',
#     'total_sulfur_dioxide',
    'ph',
    'sulphates',
    'quality',
#     'total_acidity',
    'alcohol_density',
#     'ph_acidity',
#     'ideal_ph',
    'percent_free_sulfur',
    'percent_acidity'
]]


# # Feature Selection
# 
# Refer to my work [HERE](https://www.kaggle.com/code/shilongzhuang/s3e5-training-with-data-from-uci#Feature-Selection) how I went around with my feature selection process.

# In[19]:


features = [c for c in _.columns if c not in ['quality']]
target = 'quality'


# In[20]:


# Define inputs and target feature 
X, y = _[features], _[target]

# add constant to input variables
X = sm.add_constant(X)

# Fit regression model
lr_model = sm.OLS(y, X).fit()

# Present model summary
lr_model.summary()


# In[21]:


def lgbm_objective(trial):
    
    params_optuna = {
        
        'scale_pos_weight':trial.suggest_int('scale_pos_weight', 1, 3),
        #'lambda_l1': trial.suggest_float('lambda_l1', 1e-12, 2, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 5, 25.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 35, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.65, 0.85),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.65),
        'bagging_freq': trial.suggest_int('bagging_freq', 4, 9),
         'min_child_samples': trial.suggest_int('min_child_samples', 40, 90),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 90, 150),
        "max_depth": trial.suggest_int("max_depth", 6, 12),

        'num_iterations':10000,
        'learning_rate':0.1
    }
    
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    all_scores = []
    
    for i,(train_idx, val_idx) in enumerate(cv.split(_[features],_[target])):
        X_train, X_valid = _.iloc[train_idx][features], _.iloc[val_idx][features]
        y_train , y_valid = _[target].iloc[train_idx] , _[target].iloc[val_idx]

        model = lgbm.LGBMClassifier(**params_optuna)
        
        model.fit(X_train,
                  y_train,
                  eval_set = [(X_valid, y_valid)],
                  early_stopping_rounds=50,
                  verbose=0)

        y_pred = model.predict(X_valid)
        score = cohen_kappa_score(y_valid, y_pred)
        all_scores.append(score)

    return np.mean(all_scores)


# In[22]:


# study = optuna.create_study(direction='maximize', sampler = TPESampler())
# study.optimize(func=lgbm_objective, n_trials=50)

# print(f"\tBest score: {study.best_value:.5f}")
# print(f"\tBest params:", study.best_trial.params)
# lgb_params = study.best_trial.paramsv


# In[23]:


lgb_params = {'scale_pos_weight': 3,
 'lambda_l2': 9.823545438990115,
 'num_leaves': 36,
 'feature_fraction': 0.6869830110384807,
 'bagging_fraction': 0.5586487050305629,
 'bagging_freq': 5,
 'min_child_samples': 67,
 'min_data_in_leaf': 145,
 'max_depth': 9}

lgb_preds = []
lgb_scores = []
lgb_fimp = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(cv.split( _[features], _[target])):
    
    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    
    X_train, X_valid = _.iloc[train_idx][features], _.iloc[valid_idx][features]
    y_train , y_valid = _[target].iloc[train_idx] , _[target].iloc[valid_idx]
        
    model = lgbm.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train, verbose=0)
    
    preds_valid = model.predict(X_valid)
    score = cohen_kappa_score(y_valid,  preds_valid, weights = "quadratic")
    lgb_scores.append(score)
    run_time = time.time() - start_time
    
    print(f"Fold={fold+1}, Quadratic Kappa Metric score: {score:.2f}, Run Time: {run_time:.2f}s")
    
    test_preds = model.predict(test[features])
    lgb_preds.append(test_preds)
    
print("Mean Score :", np.mean(lgb_scores))


# In[24]:


lgb_preds


# In[25]:


from scipy.stats import mode

lgb_submission = submission.copy()
lgb_submission[target] = np.squeeze(mode(np.column_stack(lgb_preds), axis = 1)[0].astype('int'))
lgb_submission.to_csv("submission.csv",index=False)

lgb_submission.head()


# In[ ]:




