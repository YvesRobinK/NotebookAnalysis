#!/usr/bin/env python
# coding: utf-8

# ---
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Binary Classification of Machine Failures</p>
# ---

# <center>
# <img src="https://gesrepair.com/wp-content/uploads/35DDEBA8-EA7C-4121-AC06-CEBA29C56D07-1024x592.jpeg" width=800>
# </center>
# 
# <center>
# <img src="https://media.istockphoto.com/id/1215001603/vector/the-robot-is-broken-and-smoking-page-404.jpg?s=612x612&w=0&k=20&c=gS3gqvddHN1Fa5dG3rGDlL-BEl4G-1947Ph6yIIiEwU=" width=300>
# </center>

# <div style="border-radius:10px;
#         border :black solid;
#         padding: 15px;
#         background-color:lightgreen;
#        font-size:110%;
#         text-align: left">
#  <span style="color:'blue' ;">This is a competition notebook for the competition <strong>Playground Series season 3 episode 17</strong><br>
# The dataset provided for the competition consists of a training dataset (train.csv) and a test dataset (test.csv). The training dataset was generated using a deep learning model trained on the Machine Failure Predictions dataset. The target variable represents machine failure (binary)<br>The feature distributions in the generated dataset are close to, but not exactly the same as the original dataset. Hence we also gonna use the original dataset.<br> The objective is to predict the probability of machine failure for each sample in the test dataset.</span>

# ---
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Table of contents</p>
# ---

# ### <font color='289C4E'><font><a class='anchor' id='top'></a>
# - [Importing Libraries & Exploring the data](#1)
# - [Exploratory Data Analysis](#2)
#     - [Check for Information Bias in Train data and Original Data](#2.1)
#     - [Linear Correlation between the features](#2.2)
#     - [Feature Distributions for train & test data](#2.3)
# - [Feature Engineering](#2.4)
# - [Predictive Analysis](#3)
#    - [1 LGBMClassifier](#5)
#     - [1.1 Optuna: Hyperparameter tunning](#5.1)
#     - [1.2 Feature Importance](#11)
#     - [1.3 Recreating the data for modelling](#12)

# ---
# 
# <a id="1"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Importing Libraries & Exploring the data</p>
# 
# ---

# In[1]:


import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import math


# <h3 align="left"> <font color='blue'>Setting Style</font></h3>

# In[2]:


sns.set_style('darkgrid')

from IPython.display import HTML, display
def set_background(color):
    script = ("var cell = this.closest('.code_cell');"
              "var editor = cell.querySelector('.input_area');"
              "editor.style.background='{}';" 
              "this.parentNode.removeChild(this)"
             ).format(color)
    display(HTML('<img src onerror="{}">'.format(script)))
set_background('#E9FDFF')


# <h3 align="left"> <font color='blue'>Loading the Data</font></h3>

# In[3]:


set_background('#E9FDFF')

train_df = pd.read_csv('/kaggle/input/playground-series-s3e17/train.csv')
original_df = pd.read_csv('/kaggle/input/machine-failure-predictions/machine failure.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e17/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s3e17/sample_submission.csv')
train_df


# <h3 align="left"> <font color='blue'>Exploring the Data</font></h3>

# In[4]:


set_background('#E9FDFF')

Df = [train_df, original_df, test_df]
names = ['Training Data', 'Original Data','Test Data']
print('Data Information')
for df,name in zip(Df,names):
    print(name)
    print(df.info())
    print('--'*30)


# <div style="border-radius:10px;
#         border :black solid;
#         padding: 10px;
#         background-color:peach;
#        font-size:120%;
#         text-align: left">
#  <span style="color:'blue' ;">All three datasets have all non null values and with the all same parameters</span>

# <div style="border-radius:10px;
#         border :black solid;
#         padding: 10px;
#         background-color:peach;
#        font-size:120%;
#         text-align: left">
#  <span style="color:'blue' ;">Here id columns seems to be irrelevant, so better to drop it.</span>

# In[5]:


set_background('#E9FDFF')
train_df.drop('id',axis=1,inplace=True)
original_df.drop('UDI',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)


# In[6]:


set_background('#E9FDFF')
desc = train_df.describe().transpose()
desc = desc.style.background_gradient()
desc


# ---
# <a id="2"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Exploratory Data Analysis</p>
# ---

# <h2 align="center"><a id="2.1"></a><font color='green'>Check for Information Bias in Train data and Original Data</font></h2>

# In[7]:


set_background('#e9ecff')
f,ax=plt.subplots(1,2,figsize=(12,10))
train_df['Machine failure'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Target class (Machine failure) in training data')
ax[0].set_ylabel('')
original_df['Machine failure'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Target class (Machine failure) in original data')
ax[1].set_ylabel('')

plt.show()


# <div style="border-radius:10px;
#         border :black solid;
#         padding: 15px;
#         background-color:lightgreen;
#        font-size:110%;
#         text-align: left">
#  <span style="color:'blue' ;"><font color='darkgreen'>As there is almost double proportion of target class(Machine failure) in both train data & original data, <br>But as there is less data of the one particular class, Hence it would be better if we merge them<br>
# Threfore let's merge the these two dataframes</font></span>

# In[8]:


set_background('#e9ecff')
train_df = pd.concat([train_df,original_df],ignore_index=True)
train_df


# In[9]:


set_background('#e9ecff')
# Drop the duplicates from the data
train_df.drop_duplicates(inplace=True)
train_df


# <h2 align="center"><a id="2.2"></a><font color='darkgreen'>Linear Correlation between the features</font></h2>

# In[10]:


train_df.columns


# In[11]:


set_background('#e9ecff')
plt.figure(figsize=(12,8))
sns.heatmap(data=train_df.drop(['Product ID', 'Type'],axis=1).corr(),annot=True,cmap='Greens')
plt.title('Correlation Matrix for Features of Train Data');
plt.show()


# <h2 align="center"><a id="2.3"></a><font color='green'>Feature Distributions for train & test data</font></h2>

# In[16]:


set_background('#e9ecff')
train_df


# In[17]:


set_background('#e9ecff')
train_df.drop('Machine failure',axis=1).columns


# In[18]:


set_background('#e9ecff')
plt.figure(figsize=(10,30))
i = 1
for col in train_df.drop('Machine failure',axis=1).columns[2:7]:
    plt.subplot(11,2,i)
    sns.histplot(x=train_df[col],color='#288BA8',kde=True,lw=1)
    plt.title("training data: distribution of '{}' feature".format(col));
   
    plt.subplot(11,2,i+1)
    sns.histplot(x=test_df[col],color='#B22222',kde=True,lw=1)
    plt.title("testing data: distribution of '{}' feature".format(col));
    i+=2
plt.tight_layout()


# In[19]:


set_background('#e9ecff')
i = 1
for col in train_df.drop('Machine failure',axis=1).columns[7:]:
    f,ax=plt.subplots(1,2,figsize=(10,15))
    train_df[col].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title("Training data: distribution of '{}' feature".format(col))
    ax[0].set_ylabel('')
    test_df[col].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
    ax[1].set_title("testing data: distribution of '{}' feature".format(col))
    ax[1].set_ylabel('')
    plt.show()
    
plt.tight_layout()


# ---
# <a id="2.4"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Feature Engineering</p>
# 
# ---

# <div style="border-radius:10px;
#         border :black solid;
#         padding: 15px;
#         background-color:lightgreen;
#        font-size:110%;
#         text-align: left">
#  <span style="color:'blue' ;">I have refer several notebooks for the feature engineering part, Hence loved this community that gives chance to learn new things and different perspectives to solve the problem in different ways.  
# So I tried to used as many important features by my own and by others references.<br>
# Thanks to all my fellow kagglers to share such valuable information</span>

# In[20]:


set_background('#b6f2fa')
train_df.head(2)


# In[21]:


set_background('#b6f2fa')
test_df.head(2)


# In[22]:


set_background('#b6f2fa')
train_df.columns


# In[24]:


set_background('#b6f2fa')
train_df['Product ID'].nunique()


# In[25]:


train_df


# <h2 align="center"> <font color='green'>Standardization for numerical labels : Numeric Scaling</font></h2>

# <font color='darkgreen'>We have intotal 5 numerical columns with different scale range of the values,<br>So we will scale this numeric features between [0,1]<br>
# Numeric scaling/feature scaling, is a preprocessing technique used to standardize or normalize the numeric features in a dataset. It involves transforming the values of numeric features to a common scale, typically between 0 and 1 or with a mean of 0 and a standard deviation of 1.</font>

# In[29]:


train_df[train_df.drop('Machine failure',axis=1).columns[2:7]]


# In[32]:


set_background('#e9ecff')

train_df[train_df.drop('Machine failure',axis=1).columns[2:7]] = MinMaxScaler().fit_transform(train_df[train_df.drop('Machine failure',axis=1).columns[2:7]])
test_df[test_df.columns[2:7]] = MinMaxScaler().fit_transform(test_df[test_df.columns[2:7]])

plt.figure(figsize=(10,30))
i = 1
for col in train_df.drop('Machine failure',axis=1).columns[2:7]:
    plt.subplot(11,2,i)
    sns.histplot(x=train_df[col],color='#288BA8',kde=True,lw=1)
    plt.title("training data: distribution of '{}' feature".format(col));
   
    plt.subplot(11,2,i+1)
    sns.histplot(x=test_df[col],color='#B22222',kde=True,lw=1)
    plt.title("testing data: distribution of '{}' feature".format(col));
    i+=2
plt.tight_layout()


# <h2 align="center"> <font color='green'>Feature Generation</font></h2>

# In[39]:


set_background('#b6f2fa')

def feat(df):
    df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    # Calculate temperature difference
    df['TemperatureDifference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    # Calculate temperature variability
    df['TemperatureVariability'] = df[['Air temperature [K]', 'Process temperature [K]']].std(axis=1)
    # Calculate temperature ratio
    df['TemperatureRatio'] = df['Process temperature [K]'] / df['Air temperature [K]']
    # Calculate tool wear rate
    df['ToolWearRate'] = df['Tool wear [min]'] / (df['Tool wear [min]'].max())
    # Calculate temperature change rate
    df['TemperatureChangeRate'] = df['TemperatureDifference'] / df['Tool wear [min]']
    # Doing this to remove outlier of infinity which is due to zero present in toolwearmin column.
    df['TemperatureChangeRate'] = np.where(df['TemperatureChangeRate']== float('inf'),1, df['TemperatureChangeRate'])
    # Calculate the total failure
    df['TotalFailures'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)
    # Torque wear ratio and product as both are dependent variables and imp for machine working
    df["TorqueWearRatio"] = df['Torque [Nm]'] / (df['Tool wear [min]'] + 0.0001)
    df["TorqueWearProduct"] = df['Torque [Nm]'] * df['Tool wear [min]']
    # Recreating the product id column 
    df["Product_id_num"] = pd.to_numeric(df["Product ID"].str.slice(start=1))
    # Feature scaling for some features to make them relevant for predictions
    features_list = ['Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    for feat in features_list:
        df[f'{feat}Squared'] = df[feat] ** 2
        df[f'{feat}Cubed'] = df[feat] ** 3
        df[f'{feat}Log'] = df[feat].apply(lambda x: math.log(x) if x > 0 else 0)
    
    for feat1 in features_list:
        for feat2 in features_list:
            df[f'{feat1}_{feat2}_Product'] = df[feat1] * df[feat2]
    # Lets just remove the unnecessory columns
    df.drop(['Product ID'],axis=1,inplace=True)
    # RNF column have the least correlation with the target variable and from the observation its not important while predicting also
    df.drop(['RNF'], axis =1, inplace = True)
    return df


# In[40]:


set_background('#b6f2fa')
train_df = feat(train_df)
test_df = feat(test_df)


# In[42]:


set_background('#b6f2fa')
test_df.head(3)


# In[43]:


set_background('#b6f2fa')
train_df = pd.get_dummies(train_df,drop_first=True)
test_df = pd.get_dummies(test_df,drop_first=True)


# In[44]:


set_background('#b6f2fa')
train_df.head(3)


# ---
# <a id="3"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Predictive Analysis</p>
# 
# ---

# <h2 align="center"> <font color='green'>Train Data</font></h2>

# In[49]:


set_background('#ccfcdf')
train = train_df.drop(['Machine failure'],axis=1)
train = pd.DataFrame(train, columns=train_df.drop(['Machine failure'],axis=1).columns)
train


# <h2 align="center"> <font color='green'>Test Data</font></h2>

# In[50]:


set_background('#ccfcdf')
test = test_df
test


# <h2 align="center"> <font color='green'>Renaming the columns</font></h2>

# ### <font color='darkgreen'>There are some problems with the column names in the dataframes <br>So lets change it to simple ones for the sake of simplicity</font>

# In[51]:


train.columns


# In[53]:


set_background('#ccfcdf')
for df in [train, test]:
    df.columns = df.columns.str.replace('[\[\]]', '', regex=True)


# <h2 align="center"> <font color='green'>Divide the Data</font></h2>

# In[56]:


set_background('#ccfcdf')
X = train
y = train_df['Machine failure']
X_test = test


# ---
# <a id="5"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">1. LGBMClassifier</p>
# ---

# In[57]:


set_background('#e3fccc')
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.metrics import roc_auc_score


# In[58]:


set_background('#e3fccc')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# ---
# <a id="5.1"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">1.1 Optuna: Hyperparameter tunning</p>
# ---

# In[59]:


set_background('#e3fccc')
get_ipython().system('pip install optuna')


# In[ ]:


set_background('#e3fccc')
import optuna
def objective(trial):
    
    n_leaves = trial.suggest_int('num_leaves', 31,100)
    max_depth = trial.suggest_int("max_depth", -1,10)
    n_estimators = trial.suggest_int("n_estimators", 10,2000)
    r_alpha = trial.suggest_int("reg_alpha", 0.0, 0.1)
    r_lambda = trial.suggest_int("reg_lambda", 0.0, 0.1)
    l_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    subsample =  trial.suggest_uniform('subsample', 0.5, 1.0)
    lambda_l1 = trial.suggest_int("lambda_l1", 0.0, 4)
    lambda_l2 = trial.suggest_int("lambda_l2", 0.0, 4)
    feature_fraction =  trial.suggest_uniform('feature_fraction', 0.5, 1.0)

    lgb = LGBMClassifier(
            num_leaves =n_leaves,
            max_depth=max_depth, 
            n_estimators=n_estimators,
            reg_alpha = r_alpha,
            reg_lambda = r_lambda,
            learning_rate = l_rate,
            subsample = subsample,
            lambda_l1 = lambda_l1,
            lambda_l2 = lambda_l2,
            feature_fraction = feature_fraction
        )
    
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

    y_pred_proba = lgb.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)


# In[65]:


set_background('#e3fccc')
trial = study.best_trial
print('Auc: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# In[66]:


set_background('#e3fccc')
lgb_clf = LGBMClassifier(**trial.params)

lgb_clf.fit(X,y)

# Predicting the probabilities of the classes using the model
pred = lgb_clf.predict_proba(X_test)
pred[:,1]


# In[67]:


set_background('#e3fccc')
# Creting DataFrame of the predicted values
df = pd.DataFrame(pred[:,1])
df.columns = ['Machine failure']
# Creating the Data for the submission to competition
sub.drop('Machine failure',axis=1,inplace=True)
sub['Machine failure']=df['Machine failure'].copy()
sub.to_csv('sub_LGBMc.csv', index=False)
sub


# ---
# <a id="11"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">1.2 Model Feature Importance</p>
# ---

# In[68]:


set_background('#cceafc')
from lightgbm import plot_importance
lgb_clf


# In[69]:


set_background('#cceafc')
plot_importance(lgb_clf, figsize=(10, 9));


# As you can see the added features are playing important role in modelling.

# ---
# <a id="12"></a>
# # <p style="padding:10px;background-color:lightgreen;border-style: solid;border-color: gray;margin:0;color:gray;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">1.3 Recreating the data for modelling</p>
# ---

# <div style="border-radius:10px;
#         border :black solid;
#         padding: 10px;
#         background-color:lightgreen;
#        font-size:110%;
#         text-align: left">
#  <span style="color:'blue' ;">So according to above feature importance analysis, <br>Some of the columns like ['Type_M','Type_L'] etc, seems to be less relevant while fitting the model, <br>so lets drop it try to remodel to predict the test data</span>

# In[70]:


X.columns


# In[71]:


set_background('#cceafc')
cols = ['Type_M','Type_L','Tool wear min_Air temperature K_Product','Process temperature KSquared','Air temperature KSquared','Torque Nm_Tool wear min_Product','Rotational speed rpm_Process temperature K_Product','Tool wear min_Process temperature K_Product','Tool wear min_Rotational speed rpm_Product','Rotational speed rpmSquared', 'Torque Nm_Air temperature K_Product','ToolWearRate','Process temperature K_Air temperature K_Product', 'Process temperature KLog','Rotational speed rpm_Torque Nm_Product','Rotational speed rpm_Air temperature K_Product','Torque NmSquared','Air temperature KLog']
train.drop(cols, axis = 1, inplace = True)
test.drop(cols, axis = 1, inplace = True)
X = train
y = train_df['Machine failure']
X_test = test


# In[72]:


set_background('#cceafc')
lgb_clf = LGBMClassifier(**trial.params)
lgb_clf.fit(X,y)

# Predicting the probabilities of the classes using the model
pred = lgb_clf.predict_proba(X_test)


# In[73]:


set_background('#cceafc')
# Creting DataFrame of the predicted values
df = pd.DataFrame(pred[:,1])
df.columns = ['Machine failure']
# Creating the Data for the submission to competition
sub.drop('Machine failure',axis=1,inplace=True)
sub['Machine failure']=df['Machine failure'].copy()
sub.to_csv('sub_LGBMc_after_feature_reduction.csv', index=False)
sub

