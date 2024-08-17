#!/usr/bin/env python
# coding: utf-8

# <blockquote style="margin-right:auto; margin-left:auto; color:white; background-color: lightseagreen; padding: 1em; margin:24px;">
# 
# <font color="white" size=+3.0><b>Summary</b></font>  
#         
# <ul>
# <li> Tried several classifiers (such as LogisticRegression, ExtraTrees and kNN, VotingClassifiers, etc..) and blended submission at older versions of the NB. However, the current best model is found to be a single Logistic regression model.
# <li> Also tried feature engineering by creating new features.
# <li> Only used 12 features (6 original and 6 derived). Going forward <strong> this might change with further experiments</strong>.          
#    
#     
# </ul>  
#     
# <strong>Note:</strong>
#     
# At the end of the notebook I will use/try a new kind of ML library called feyn/QLattice by ABZU, not as a means to boost performance but to try out the library. It was on my to-do list ever since I saw example notebooks (see references) but was not able to put it in action up until now. 
# 
# </blockquote>
# 
# #### Import Libararies

# In[1]:


# all may not be needed

import os
import sys
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pylab as plt
from colorama import Fore, Back, Style


from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression,HuberRegressor
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier,VotingClassifier,StackingClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import cross_validate, StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer, LabelEncoder, StandardScaler, MinMaxScaler

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')


# ### 1. Dataset Overview

# In[3]:


train.head(3)


# In[4]:


test.head(3)


# #### Data size
# - Train dataset  has 26570 rows and 25 columns including the target column (failure).
# - Test dataset has 20775 rows and 24 columsn.

# In[5]:


display(train.shape)
display(test.shape)


# #### Null values
# - Around 3% of the data (cells) is missing in both train and test datset.
# - We will need to impute.

# In[6]:


print('Train data missing value is = {} %'.format(100* train.isna().sum().sum()/(len(train)*25)))
print('Test data missing value is  = {} %'.format(100* test.isna().sum().sum()/(len(test)*25)))


# In[7]:


train_na_cols = [col for col in train.columns if train[col].isnull().sum()!=0]
print('Train data cols with missing values ares: \n', train_na_cols)

print('\n')

test_na_cols = [col for col in test.columns if test[col].isnull().sum()!=0]
print('Train data cols with missing values ares: \n', test_na_cols)


# In[8]:


get_ipython().system('pip install missingno')


# In[9]:


import missingno as msno
msno.matrix(train.drop('id', axis=1), color=(0.55, 0.75, 0.85), figsize=(16, 8))


# #### Correlation plots
# - Plotted the correlation heatmap to get a global idea of what features might be related to each other and may lead us to some feature engineering.
# - We see two groups where feature-to-feature correlation might exist: 
# 
# > `measurement_17` seems to be correlated with `measurements_5 and 8`
# 
# > There seem to be also some sort of correlation between `attributes_2, 3` and `measurements_0, 1`
# 
# - There could be an opportunity for a new feature to be derived from these.
# 

# In[10]:


corr = train.drop('id', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(12, 12), facecolor='#EAECEE')
cmap = sns.color_palette("rainbow", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1., center=0, annot=False,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.75})

ax.set_title('Correlation heatmap', fontsize=24, y= 1.05)
colorbar = ax.collections[0].colorbar


# In[11]:


target = train.pop('failure')
data = pd.concat([train, test])
train.shape,test.shape


# ### 2. Pre-processing/FE
# - Credits to the awesome notebook of Sawaimilert https://www.kaggle.com/code/takanashihumbert/tps-aug22-lb-0-59013
# 
# **Notes:**
# - Tried different values of `n_neighbors` for the KNNImpute and found `n_neighbors=3` to be the best 
# - I also tries other function such as `IterativeImputer` and `LGBMImputer` but `KNNImputer` was better than the rest.
# - A new feature called `area` is created by multiplying `attribute_3` and `attrinute_3`.
# - Measuerements 3 to 17 look very similar and are averaged to one new features, `measurement_avg`.
# - Two other features are derived from [Ambros' notebook](https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense).  (`m3_missing` and `m5_missing`)

# In[12]:


data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
data['area'] = data['attribute_2'] * data['attribute_3']

feature = [f for f in test.columns if f.startswith('measurement') or f=='loading']

# dictionary of dictionaries (for the 11 best correlated measurement columns), 
# we will use the dictionaries below to select the best correlated columns according to the product code)
# Only for 'measurement_17' we make a 'manual' selection :

full_fill_dict ={}
full_fill_dict['measurement_17'] = {
    'A': ['measurement_5','measurement_6','measurement_8'],
    'B': ['measurement_4','measurement_5','measurement_7'],
    'C': ['measurement_5','measurement_7','measurement_8','measurement_9'],
    'D': ['measurement_5','measurement_6','measurement_7','measurement_8'],
    'E': ['measurement_4','measurement_5','measurement_6','measurement_8'],
    'F': ['measurement_4','measurement_5','measurement_6','measurement_7'],
    'G': ['measurement_4','measurement_6','measurement_8','measurement_9'],
    'H': ['measurement_4','measurement_5','measurement_7','measurement_8','measurement_9'],
    'I': ['measurement_3','measurement_7','measurement_8']
}


# collect the name of the next 10 best measurement columns sorted by correlation (except 17 already done above):
col = [col for col in test.columns if 'measurement' not in col]+ ['loading','m3_missing','m5_missing']
a = []
b =[]
for x in range(3,17):
    corr = np.absolute(data.drop(col, axis=1).corr()[f'measurement_{x}']).sort_values(ascending=False)
    a.append(np.round(np.sum(corr[1:4]),3)) # we add the 3 first lines of the correlation values to get the "most correlated"
    b.append(f'measurement_{x}')
c = pd.DataFrame()
c['Selected columns'] = b
c['correlation total'] = a
c = c.sort_values(by = 'correlation total',ascending=False).reset_index(drop = True)
print(f'Columns selected by correlation sum of the 3 first rows : ')
display(c.head(10))

for i in range(10):
    measurement_col = 'measurement_' + c.iloc[i,0][12:] # we select the next best correlated column 
    fill_dict ={}
    for x in data.product_code.unique() : 
        corr = np.absolute(data[data.product_code == x].drop(col, axis=1).corr()[measurement_col]).sort_values(ascending=False)
        measurement_col_dic = {}
        measurement_col_dic[measurement_col] = corr[1:5].index.tolist()
        fill_dict[x] = measurement_col_dic[measurement_col]
    full_fill_dict[measurement_col] =fill_dict
    
feature = [f for f in data.columns if f.startswith('measurement') or f=='loading']
nullValue_cols = [col for col in train.columns if train[col].isnull().sum()!=0]
    
for code in data.product_code.unique():
    total_na_filled_by_linear_model = 0
    print(f'\n-------- Product code {code} ----------\n')
    print(f'filled by linear model :')
    for measurement_col in list(full_fill_dict.keys()):
        tmp = data[data.product_code==code]
        column = full_fill_dict[measurement_col][code]
        tmp_train = tmp[column+[measurement_col]].dropna(how='any')
        tmp_test = tmp[(tmp[column].isnull().sum(axis=1)==0)&(tmp[measurement_col].isnull())]

        model = HuberRegressor(epsilon=1.9)
        model.fit(tmp_train[column], tmp_train[measurement_col])
        data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data[measurement_col].isnull()),measurement_col] = model.predict(tmp_test[column])
        print(f'{measurement_col} : {len(tmp_test)}')
        total_na_filled_by_linear_model += len(tmp_test)
        
    # others NA columns:
    NA = data.loc[data["product_code"] == code,nullValue_cols ].isnull().sum().sum()
    model1 = KNNImputer(n_neighbors=3)
    #model1 = LGBMImputer(n_iter=50)
    #model1 = IterativeImputer(random_state=0) 
    data.loc[data.product_code==code, feature] = model1.fit_transform(data.loc[data.product_code==code, feature])
    print(f'\n{total_na_filled_by_linear_model} filled by linear model ') 
    print(f'{NA} filled by KNN ')
    
data['measurement_avg'] = data[[f'measurement_{i}' for i in range(3, 17)]].mean(axis=1)


# In[13]:


print("Missing values in the combined dataset after pre-peocessing is: ", format(data.isna().sum().sum()))


# #### Helper function for scaling
# 
# - Standard scaler:  source code https://www.kaggle.com/code/takanashihumbert/tps-aug22-lb-0-59013

# In[14]:


def _scale(train_data, val_data, test_data, feats):
    scaler = StandardScaler()
       
    scaled_train = scaler.fit_transform(train_data[feats])
    scaled_val = scaler.transform(val_data[feats])
    scaled_test = scaler.transform(test_data[feats])
    
    #back to dataframe
    new_train = train_data.copy()
    new_val = val_data.copy()
    new_test = test_data.copy()
    
    new_train[feats] = scaled_train
    new_val[feats] = scaled_val
    new_test[feats] = scaled_test
    
    assert len(train_data) == len(new_train)
    assert len(val_data) == len(new_val)
    assert len(test_data) == len(new_test)
    
    return new_train, new_val, new_test


# - Separate the train and test dataset

# In[15]:


train = data.iloc[:train.shape[0],:]
test = data.iloc[train.shape[0]:,:]
print(train.shape, test.shape)

groups = train.product_code
X = train
y = target


# In[16]:


# library for coding string values #Thanks to @MAXSARMENTO:
get_ipython().system(' pip install feature_engine')
from feature_engine.encoding import WoEEncoder


# In[17]:


woe_encoder = WoEEncoder(variables=['attribute_0'])
woe_encoder.fit(train, y)
X = woe_encoder.transform(train)
test = woe_encoder.transform(test)


# #### Additional features
# - These two new feaures came from [this discusion post.](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/343368)

# In[18]:


X['measurement(3*5)'] = X['measurement_3'] * X['measurement_5']
test['measurement(3*5)'] = test['measurement_3'] * test['measurement_5']

X['missing(3*5)'] = X['m5_missing'] * (X['m3_missing'])
test['missing(3*5)'] = test['m5_missing'] * (test['m3_missing'])


# In[19]:


select_feature = [
    'loading',
    'attribute_0',
    'measurement_17',
    'measurement_0',
    'measurement_1',
    'measurement_2',
    'area',
    'm3_missing', 
    'm5_missing',
    'measurement_avg',
    'measurement(3*5)',
    'missing(3*5)',      
]


# ### 3. Model/LogisticRegression
# - 5 fold CV model using a linear model (LR)
# - We see that `loading` and `measurement_17` are the top two important features for our model. 
# - Note also that the newly created features i.e, `m3_missing`, `m5_missing`, and `missing(3*5)` are among the top 10 important features.

# In[20]:


lr_oof = np.zeros(len(train))
lr_test = np.zeros(len(test))
lr_auc = 0
importance_list = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("Fold:", fold_idx+1)
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    x_train, x_val, x_test = _scale(x_train, x_val, test, select_feature)
    
    model = LogisticRegression(max_iter=200, C=0.0001, penalty='l2', solver='newton-cg')
    model.fit(x_train[select_feature], y_train)
    importance_list.append(model.coef_.ravel())

    val_preds = model.predict_proba(x_val[select_feature])[:, 1]
    print("FOLD: ", fold_idx+1, " ROC-AUC:", round(roc_auc_score(y_val, val_preds), 5))
    lr_auc += roc_auc_score(y_val, val_preds) / 5
    lr_test += model.predict_proba(x_test[select_feature])[:, 1] / 5
    lr_oof[val_idx] = val_preds

print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}{Style.RESET_ALL}")
print(f"{Fore.BLUE}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof), 5)}{Style.RESET_ALL}")

importance_list.append(model.coef_.ravel())
importance_df = pd.DataFrame(np.array(importance_list).T, index=x_train[select_feature].columns)
importance_df['mean'] = importance_df.mean(axis=1).abs()
importance_df['feature'] = x_train[select_feature].columns
importance_df = importance_df.sort_values('mean', ascending=True).reset_index()

fig, ax = plt.subplots(figsize=(12, 8), facecolor='#EAECEE')
plt.barh(importance_df.index, importance_df['mean'], color='lightseagreen')

plt.yticks(ticks=importance_df.index, labels=importance_df['feature'])
plt.title('LogisticRegression feature importance', fontsize=20, y= 1.05)
plt.show()


# ### 4. Submission
# 

# In[21]:


sub = pd.DataFrame({'id': submission.id, 'failure': lr_test/5})
sub.to_csv("submission.csv", index=False)


# In[22]:


sub.head()


# ___
# ## Modeling using `QLattice`
# 
# **Ref**: https://docs.abzu.ai/docs/guides/getting_started/qlattice.html
# 
# **The QLattice**
# 
# "The QLattice is a supervised machine learning tool for symbolic regression, and is a technology developed by **Abzu** that is inspired by Richard Feynman's path integral formulation. That's why we've named our python library `Feyn`, and the **Q** in `QLattice` is for **Quantum**.
# 
# "It composes functions together to build mathematical models betweeen the inputs and output in your dataset. The functions vary from elementary ones such as *addition*, *multiplication* and *squaring*, to more complex ones such as *natural logarithm*, *exponential* and *tanh*.
# 
# "Overall, symbolic regression approaches tend to keep a high performance, while still maintaining generalisability, which separates it from other popular models such as random forests. Our own benchmarks show similar results."
# 
# **Feyn**
# 
# "Feyn is the python module for using the QLattice, and training models that have been sampled from the QLattice.
# 
# When sampling models, you define criteria in Feyn that these models must meet. Some examples include: is it a classification or regression problem, which features you want to learn about, what functions you want to include, how complex the models may be, and other such constraints."" 

# In[23]:


get_ipython().system('pip install feyn')


# In[24]:


import feyn
from sklearn.model_selection import train_test_split


# In[25]:


df = pd.concat([X, y], axis=1)
df.drop('id', axis=1).head(3)


# ### Model Training 
# - (1) Splitting data in train, validation

# In[26]:


output = 'failure'
train, test_ = train_test_split(df,
                               stratify=df[output], 
                               train_size=0.8, 
                               random_state=42)


# - (2) Connect to the community Qlattice

# In[27]:


ql = feyn.connect_qlattice()


# In[28]:


ql.reset(random_seed=42)


# - (3) Declare our `categorical` features

# In[29]:


stypes = {
    'product_code': 'cat',
    'attribute_1': 'cat'
}


# - (4) Fitting the models

# In[30]:


models = ql.auto_run(df, 
                     output_name=output, 
                     kind="classification", 
                     stypes= stypes,
                     loss_function='binary_cross_entropy',
                     max_complexity=10, 
                     n_epochs=10)


# **Note:** We also have the option to print out and see the transfer function(s) the best models use to arrive at the prediction they have made.

# In[31]:


sym_model= models[0].sympify()
sym_model.as_expr()


# ### Model Evaluation
# - To help evaluate our model, we can code in `best_model.plot(train, test)` and print/plot the train/test metrics, confusion matrix and so on. 
# - Note that our best model here is models[0]. QLattice calculate, rank and store (sorted in acsending order) our models based on the `loss_function` we provide. So the our best model (with lowest loss) is models[0] (the first model in the list of 10 of the top models)
# - Just like the other models we have already seen, the `AUC` score is not so greate. The dataset we have is quite noisey. But we know that already.

# In[32]:


models[0].plot(train,test_)


# In[33]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))
models[0].plot_probability_scores(train, title='Total probability distribution', ax = ax)
plt.show()


# ### Feature Importance
# - We already have seen that the model used only two features for its final prediction. So feature importance is not necessary in that regard.
# - However, we can use it to further inspect individual predictions and how they contributed to the SHAP values of the features (important features). For example the table below shows the degree to which each feature contributed towards the negative (False) or positive direction (True) for the  prediction. 
# 

# In[34]:


def filter_FN(prediction_series, truth_series, truth_df):
    '''filters and returns the false negatives in predicted dataframe
    '''
    pred = prediction_series < 0.5
    return truth_df.query('@truth_series == @pred and @truth_series == True')

predictions = models[0].predict(test_)
false_negatives = filter_FN(predictions, test_.failure, test_)
false_negatives.shape


# In[35]:


from feyn.insights import FeatureImportanceTable
features = models[0].features
table = FeatureImportanceTable(models[0], 
                               false_negatives[features], 
                               bg_data=train[features], 
                               max_rows = 15,
                               style='fill')# the other option for style is 'bar'
table


# In[36]:


import shap
shap_values = table.importance_values

shap.summary_plot(shap_values, feature_names=models[0].features)


# #### Personal note on feyn/QLattice:
# - I found the library to be easy to use and an interactive ML toolbox.
# - With ‘minimal’ code one can get a lot of information about the model outputs.
# - It can also be used as a good `feature selection` tool. The best model used only `loading` and `measuremsnt_17` for its prediction; meaning that the model *thinks* that these two are the most important features - which is in agreement with our `logisticRegression` model's feature importance output. 
# - Performance wise it looks on par with the household ML algo. Actually `feyn` has a model comparison function and is easy to compare its performance with the usual models such random forest, LR, etc. [`refer notebook 3 below to see an example`]
# - If you are curious, I encourage you to try it for yourself. You may want to refer to the notebooks below. Chris and Casper work for the company that develops feyn, ABZU.
# 

# #### Further reference on QLattice (kaggle notebooks)
# 1. [QLattice in Optiver Market Volatility](https://www.kaggle.com/code/mpwolke/qlattice-in-optiver-market-volatility): by Marilia Prata. This notebook introduced me to QLattice for the first time.
# 2. [QLattice on diabetic predictions](https://www.kaggle.com/code/chriscave/qlattice-on-diabetic-predictions): by Chris Cave
# 3. [Explainable model for diabetes using the QLattice](https://www.kaggle.com/code/wilstrup/explainable-model-for-diabetes-using-the-qlattice): by Casper Wilstrup
# 4. [The QLattice shows how 3 features predict toxicity](https://www.kaggle.com/code/livtoft/the-qlattice-shows-how-3-features-predict-toxicity): by Liv Toft
# 
# 
# 
# ### ___END OF NOTEBOOK.___
# 
# 

# In[ ]:




