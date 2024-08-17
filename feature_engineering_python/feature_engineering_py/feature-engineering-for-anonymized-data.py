#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering for Anonymized Data:

# #### This notebook will show some ideas for feature engineering for anonimized data:

# # Import Libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# In[2]:


# Competition Metric (Thanks to @validmodel : https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/409691)
def balanced_log_loss(y_true, y_pred):
    N = len(y_true)
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15) # clip y_pred to avoid taking log of 0 or 1
    loss = -(1/N) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return loss


# # Read Data:

# In[3]:


train_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
test_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")


# In[4]:


ID = test_df['Id']
train_df.drop('Id',inplace=True,axis=1)
test_df.drop('Id',inplace=True,axis=1)


# In[5]:


SUBMIT = True


# # Fit Baseline:

# In[6]:


# Label Encoding:
feats = list(train_df.select_dtypes(include=['object','category']).columns)
le = LabelEncoder()
df = pd.concat([train_df, test_df])
for f in feats:
    le.fit(df[f])
    train_df[f] = le.transform(train_df[f])
    test_df[f] = le.transform(test_df[f])


# In[7]:


if not SUBMIT:
    X = train_df.copy()
    y = X['Class']
    X.drop('Class',inplace=True,axis=1)

    cb = CatBoostClassifier(depth=6, random_state=42, verbose=0)
    cb.fit(X,y);


# # 1- Feature Importance: 
# Here we use the feature importance of the Catboost to know which features are really important for predicting the target. More reliable approaches such as LOFO or Permutation Importance are recommended. (Have a look at LOFO from @aerdem4 : https://www.kaggle.com/code/aerdem4/icr-lofo-feature-importance):

# In[8]:


#Plot the Features Importances
def plotImp(model, X , num = 20, fig_size = (40, 20)):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    plt.title('Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('importances-01.png')
    plt.show()
    sns.set()  


# In[9]:


if not SUBMIT:
    plotImp(cb,X)


# You can add intearctions or aggregations for the top features.  

# # 2- Correlation: 
# In addition to the top important features, we can try to do some feature engineering for the features that are high correlated to the target or the top important features:

# In[10]:


if not SUBMIT:
    corr_matrix = train_df.corr()
    print(corr_matrix["Class"].sort_values(ascending=False))


# Interestingly, The features that are high (negative or positive) correlated to the target are the most important to the target in Catboost.

# # 3- Unique Values: 
# We can add aggregations for the features in 1 and 2 based on the features that have low number of unique values:

# In[11]:


if not SUBMIT:
    #The unique values of each feature
    cat_cols = train_df.columns
    for col in cat_cols:
        print(col, train_df[col].nunique())


# For example, we can see that the features ['EJ','BN','DV'] have low number of unique values. So we can use them to aggregate the top important features.

# # 4- Target Encoding:

# Aggregations for the target based on low cadinality features:

# In[12]:


def Agg(Feature):
    for dataset in (train_df,test_df):
        for feat_1 in ['EJ','BN','DV']:
            dataset[f'{Feature}_Agg_{feat_1}_mean'] = dataset[feat_1].map(dict(train_df.groupby(feat_1)[Feature].mean()))
            dataset[f'{Feature}_Agg_{feat_1}_std'] = dataset[feat_1].map(dict(train_df.groupby(feat_1)[Feature].std()))
            dataset[f'{Feature}_Agg_{feat_1}_median'] = dataset[feat_1].map(dict(train_df.groupby(feat_1)[Feature].median()))        
Agg('Class')


# # Modeling: 
# Let's use 5 folds stratified CV because we have imbalanced target and use Catboost as a model given that it does great job with low number of samples:

# In[13]:


cb_params = {'depth': 6, 'iterations': 1000, 'learning_rate': 0.01, 'verbose':0, 'task_type':'CPU'}
cb = CatBoostClassifier(**cb_params, random_state=42)


# In[14]:


if not SUBMIT:
    print('Validating...')

    X = train_df.drop('Class',axis=1)
    y = train_df['Class'].values

    scores = []                   
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_Train, X_Test = X.loc[train_index,:], X.loc[test_index,:]
        y_Train, y_Test = y[train_index], y[test_index]
        cb.fit(X_Train,y_Train)
        y_pred = cb.predict_proba(X_Test)[:,1]
        scores.append(balanced_log_loss(y_Test,y_pred))
        print(scores[-1])

    print("\nMean:",np.mean(scores),"\nSTD: ", np.std(scores))


# In[15]:


if not SUBMIT:
    plotImp(cb,X)


# # Inference:

# In[16]:


if SUBMIT:
    X = train_df.drop('Class',axis=1)
    y = train_df['Class'].values
    cb.fit(X,y)
    Predictions = cb.predict_proba(test_df)
    submission = pd.DataFrame({"Id": ID })
    submission[['class_0','class_1']] = Predictions
    submission.to_csv('submission.csv',index=False)

