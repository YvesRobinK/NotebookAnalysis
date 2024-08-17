#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn import metrics

from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install optuna')
import optuna

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
get_ipython().system('pip3 install catboost')
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
import time


# In[2]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')
print(df_train.columns)
train_ids=df_train['passengerid'].values
print(df_train.shape)
print(df_train.info())

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_test.columns = df_test.columns.str.lower().str.replace(' ', '_')
test_ids=df_test['passengerid'].values
print(df_test.shape)
print(df_test.info())

df_train.sample(30)


# In[3]:


def null_count_with_percent(df):
  total = df.isnull().sum().sort_values(ascending=False)
  percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
  missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  return missing_data


# In[4]:


print(null_count_with_percent(df_train))
print(null_count_with_percent(df_test))


# **i'll drop capin because the null values are more than 40%**

# In[5]:


df_train=df_train.drop(columns='cabin')
df_test=df_test.drop(columns='cabin')


# **handle age null values**

# In[6]:


ImputedModule = SimpleImputer(strategy='mean').set_output(transform="pandas")
X=df_train['age'].values.reshape(-1, 1)
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
df_train['age']=X

ImputedModule = SimpleImputer(strategy='mean').set_output(transform="pandas")
X=df_test['age'].values.reshape(-1, 1)
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
df_test['age']=X


# **handle embarked null values on train data**

# In[7]:


print(df_train.embarked.value_counts())


# In[8]:


ImputedModule = SimpleImputer(strategy='constant', fill_value='messing').set_output(transform="pandas")
X=df_train['embarked'].values.reshape(-1, 1)
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
df_train['embarked']=X


# **handle fare null on test data**

# In[9]:


print(df_test.fare.value_counts())


# In[10]:


ImputedModule = SimpleImputer(strategy='mean').set_output(transform="pandas")
X=df_test['fare'].values.reshape(-1, 1)
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
df_test['fare']=X


# In[11]:


print(null_count_with_percent(df_train))
print(null_count_with_percent(df_test))


# **no more null data !!**

# # **EDA**

# In[12]:


target='survived'


# In[13]:


class_counts = df_train[target].value_counts()
class_proportions = class_counts / df_train.shape[0]
class_proportions = class_proportions.values.tolist()
class_proportions_str = [f'{prop:.2%}' for prop in class_proportions]

# Set the color palette
colors = sns.color_palette('pastel')[0:len(class_counts)]

# Plot the distribution of the target variable
plt.figure(figsize=(8, 4))
sns.countplot(x=target, data=df_train, palette=colors)
plt.title('Distribution of Target Variable', fontsize=16)
plt.xlabel(target, fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.ylim([0, len(df_train)])
for i, count in enumerate(class_counts):
    plt.text(i, count + 50, class_proportions_str[i], ha='center', fontsize=14, color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.show()


# In[14]:


def survived_bar_plot(feature):
    plt.figure(figsize = (6,4))
    sns.barplot(data = df_train , x = feature , y = target).set_title(f"{feature}  ")
    plt.show()
    
def survived_table(feature):
    return df_train[[feature, target]].groupby([feature], as_index=False).mean().sort_values(by=target, ascending=False).style.background_gradient(low=0.75,high=1)

def survived_hist_plot(feature):
    plt.figure(figsize = (10,10))
    sns.histplot(data = df_train , x = feature , hue = target,binwidth=5,palette = sns.color_palette(["yellow" , "green"]) ,multiple = "stack" ).set_title(f"{feature} Vs ")
    plt.show()


# In[15]:


df_train.columns


# In[16]:


survived_bar_plot('sex')


# In[17]:


survived_table('sex')


# In[18]:


survived_bar_plot('pclass')


# In[19]:


survived_table('pclass')


# In[20]:


survived_bar_plot('embarked')


# In[21]:


survived_table('embarked')


# In[22]:


survived_bar_plot('sibsp')


# In[23]:


survived_table('sibsp')


# In[24]:


survived_bar_plot('parch')


# In[25]:


survived_table('parch')


# In[26]:


survived_hist_plot('fare')


# In[27]:


survived_hist_plot('age')


# **age vs pclass vs target**

# In[28]:


plot , ax = plt.subplots(1 , 3 , figsize=(14,4))
sns.histplot(data = df_train.loc[df_train["pclass"]==1] , x = "age" , hue = target,binwidth=5,ax = ax[0],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("1-Pclass")
sns.histplot(data = df_train.loc[df_train["pclass"]==2] , x = "age" , hue = target,binwidth=5,ax = ax[1],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("2-Pclass")
sns.histplot(data = df_train.loc[df_train["pclass"]==3] , x = "age" , hue = target,binwidth=5,ax = ax[2],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("3-Pclass")
plt.show()


# In[29]:


plot , ax = plt.subplots(1 ,2 , figsize=(14,4))
sns.histplot(data = df_train.loc[df_train["sex"]=='male'] , x = "age" , hue = target,binwidth=5,ax = ax[0],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("1-male")
sns.histplot(data = df_train.loc[df_train["sex"]=='female'] , x = "age" , hue = target,binwidth=5,ax = ax[1],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("2-female")
plt.show()


# # **Feature Engineering**

# In[30]:


df_train['family_size'] = df_train['sibsp'] + df_train['parch'] + 1
df_test['family_size'] = df_test['sibsp'] + df_test['parch'] + 1


# In[31]:


survived_table('family_size')


# In[32]:


df_train['is_alone'] = df_train['family_size'].map(lambda x: 1 if x == 1 else 0).astype(bool)
df_train['family_small'] = df_train['family_size'].map(lambda x: 1 if x == 2 else 0)
df_train['family_med'] = df_train['family_size'].map(lambda x: 1 if 3 <= x <= 4 else 0)
df_train['family_big'] = df_train['family_size'].map(lambda x: 1 if x >= 5 else 0)



df_test['is_alone'] = df_test['family_size'].map(lambda x: 1 if x == 1 else 0).astype(bool)
df_test['family_small'] = df_test['family_size'].map(lambda x: 1 if x == 2 else 0)
df_test['family_med'] = df_test['family_size'].map(lambda x: 1 if 3 <= x <= 4 else 0)
df_test['family_big'] = df_test['family_size'].map(lambda x: 1 if x >= 5 else 0)


df_train=df_train.drop(columns=['family_size','parch','sibsp'])
df_test=df_test.drop(columns=['family_size','parch','sibsp'])



# In[33]:


df_train['title']=df_train.name.str.extract(' ([A-Za-z]+)\.', expand=False)
title_names = (df_train['title'].value_counts() < 5)
df_train['title'] = df_train['title'].apply(lambda x: 'rare' if title_names.loc[x] == True else x)

df_test['title']=df_test.name.str.extract(' ([A-Za-z]+)\.', expand=False)
title_names = (df_test['title'].value_counts() < 5)
df_test['title'] = df_test['title'].apply(lambda x: 'rare' if title_names.loc[x] == True else x)


survived_table('title')


# In[34]:


df_train['is_mrs'] = df_train['title'].map(lambda x: 1 if x == 'Mrs' else 0)
df_train['is_miss'] = df_train['title'].map(lambda x: 1 if x == 'Miss' else 0)
df_train['is_master'] = df_train['title'].map(lambda x: 1 if x == 'Master' else 0)



df_test['is_mrs'] = df_test['title'].map(lambda x: 1 if x == 'Mrs' else 0)
df_test['is_miss'] = df_test['title'].map(lambda x: 1 if x == 'Miss' else 0)
df_test['is_master'] = df_test['title'].map(lambda x: 1 if x == 'Master' else 0)


df_train=df_train.drop(columns=['title','name'])
df_test=df_test.drop(columns=['title','name'])




# In[35]:


df_train=df_train.drop(columns=['ticket'])
df_test=df_test.drop(columns=['ticket'])

df_train


# In[36]:


df_train['fare_band'] = pd.qcut(df_train['fare'], 4)
df_train[['fare_band', target]].groupby(['fare_band'], as_index=False).mean().sort_values(by='fare_band', ascending=False)


# In[37]:


df_train.loc[ df_train['fare'] <= 7.91, 'fare'] = 0
df_train.loc[(df_train['fare'] > 7.91) & (df_train['fare'] <= 14.454), 'fare'] = 1
df_train.loc[(df_train['fare'] > 14.454) & (df_train['fare'] <= 31), 'fare']   = 2
df_train.loc[ df_train['fare'] > 31, 'fare'] = 3
df_train['fare'] = df_train['fare'].astype(int)



df_test.loc[ df_test['fare'] <= 7.91, 'fare'] = 0
df_test.loc[(df_test['fare'] > 7.91) & (df_test['fare'] <= 14.454), 'fare'] = 1
df_test.loc[(df_test['fare'] > 14.454) & (df_test['fare'] <= 31), 'fare']   = 2
df_test.loc[ df_test['fare'] > 31, 'fare'] = 3
df_test['fare'] = df_test['fare'].astype(int)


df_train=df_train.drop(columns=['fare_band'])
df_train


# In[38]:


df_train['age_band'] = pd.cut(df_train['age'], 5)
df_train[['age_band', target]].groupby(['age_band'], as_index=False).mean().sort_values(by='age_band', ascending=True)


# In[39]:


df_train.loc[ df_train['age'] <= 16, 'age'] = 0
df_train.loc[(df_train['age'] > 16) & (df_train['age'] <= 32), 'age'] = 1
df_train.loc[(df_train['age'] > 32) & (df_train['age'] <= 48), 'age'] = 2
df_train.loc[(df_train['age'] > 48) & (df_train['age'] <= 64), 'age'] = 3
df_train.loc[ df_train['age'] > 64, 'age']

df_test.loc[ df_test['age'] <= 16, 'age'] = 0
df_test.loc[(df_test['age'] > 16) & (df_test['age'] <= 32), 'age'] = 1
df_test.loc[(df_test['age'] > 32) & (df_test['age'] <= 48), 'age'] = 2
df_test.loc[(df_test['age'] > 48) & (df_test['age'] <= 64), 'age'] = 3
df_test.loc[ df_test['age'] > 64, 'age']

df_train=df_train.drop(columns=['age_band'])


# In[40]:


df_train.to_csv('train_backup.csv')
df_test.to_csv('test_backup.csv')


# **One Hot Encoding**

# In[41]:


cat_feat=['sex','embarked','pclass']

cats_encoded=[]

for cat in cat_feat:
  df_train[cat]= df_train[cat].astype('category')
  df_train[cat+'_encoded'] = df_train[cat].cat.codes
    
  df_test[cat]= df_test[cat].astype('category')
  df_test[cat+'_encoded'] = df_test[cat].cat.codes
    
  cats_encoded.append(cat+"_encoded")


one_hot_encoded_data = pd.get_dummies(df_train[cats_encoded], columns =cats_encoded)
df_train[one_hot_encoded_data.columns]=one_hot_encoded_data.values
df_train=df_train.drop(columns=cat_feat)
df_train=df_train.drop(columns=cats_encoded)

one_hot_encoded_data_test = pd.get_dummies(df_test[cats_encoded], columns =cats_encoded)
df_test[one_hot_encoded_data_test.columns]=one_hot_encoded_data_test.values
df_test=df_test.drop(columns=cat_feat)
df_test=df_test.drop(columns=cats_encoded)



print(df_train.shape)
print(df_train.columns)

print(df_test.shape)
print(df_test.columns)


# In[42]:


df_train.embarked_encoded_3.value_counts()


# **embarked_encoded_3 in train data is the null values i handled ,ill drop it**

# In[43]:


df_train=df_train.drop(columns='embarked_encoded_3')

print(df_train.shape)

print(df_test.shape)


# In[44]:


df_train


# **Correlation between data and target**

# In[45]:


plt.figure(figsize=(15,13))
cor = df_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[46]:


# selected 
df_tra=df_train.copy()


# **VERY GOOD !! I have good correlations with the target**

# # **Split data to train and test**

# In[47]:


X=df_train[df_train.columns.difference(['passengerid',target])].values
print(X.shape)
Y=df_train[target].values
print(Y.shape)


# # Modelling

# In[48]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,roc_auc_score

param ={'lambda': 1.2475525176000515,
        'learning_rate': 0.05,
        "objective": "binary:logistic",
        'colsample_bytree': 0.9,
        'subsample': 1.0, 
        'max_depth': 2,
        'min_child_weight': 3,
        "eval_metric": "auc"
        
       }




def xgboost_bin_classif(X, y, test_size=0.2, random_state=42, num_round=100):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]


    num_round = 5000
    evals_result = {}
    bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round,
                    evals_result=evals_result,
                    maximize=False, obj=None,
                    evals=watchlist, verbose_eval=200)
    y_pred = bst.predict(dtest)
    test_auc = roc_auc_score(y_test, y_pred)
    print(f"auc: {test_auc}")
    percentiles = np.arange(1, 100, 1)
    base_rate = (y == 1).sum() / y.shape[0]
    
    
    results = []
    for i,percentile in enumerate(percentiles):
        cutoff = np.percentile(y_pred, percentile)
        predictions = np.where(y_pred >= cutoff, 1, 0)
        cm = confusion_matrix(y_test, predictions, labels=[0,1]).copy()

        tp = cm[0,0]
        precision = tp / cm[:,0].sum()
        recall = tp / cm[0,:].sum()
        accuracy = cm.diagonal().sum() / cm.sum()
        f1score = 2 * (precision * recall) / (precision + recall)
        lift = precision/base_rate

        res = dict(
            percentile=percentile,
            cutoff = cutoff,
            precision=precision,
            recall=recall,
            lift=lift,
            f1score=f1score,
            accuracy=accuracy,
        )

        results.append(res)
    
    results = pd.DataFrame(results)
    cut_off=0.415098
    predictions = np.where(y_pred > cut_off, 1, 0)
    cm = confusion_matrix(y_test, predictions, labels=[1,0])

    # df_result = pd.DataFrame({
    # 'target': y_test,
    # 'predected': y_pred
    #  })




    return bst,test_auc,results,cm 


# In[49]:


bst,auc,results,cm=xgboost_bin_classif(X,Y)


# In[50]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[1,0])
disp.plot()


# In[51]:


x = results[results.percentile.isin( np.arange(0, 100, 5) )]
x.set_index("percentile")


# In[52]:


submission_test =bst.predict(xgb.DMatrix(df_test[df_test.columns.difference(['passengerid'])].values))


# In[53]:


cut_off=0.415098
predictions = np.where(submission_test > cut_off, 1, 0)


# In[54]:


pd.Series(predictions).value_counts()


# # Submission

# In[55]:


print(predictions.shape)

sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# Add predictions
sub['Survived']=predictions

sub.to_csv('submission.csv', index=False)

