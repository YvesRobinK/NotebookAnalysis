#!/usr/bin/env python
# coding: utf-8

# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
# <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet">
# 
# ![olisa-obiora-ToRz-jwncrM-unsplash (1).jpg](attachment:2375099b-7b97-409b-af1f-c7a3046914cd.jpg)
# 
# <h1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;"><b>Introduction</b></h1>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Welcome to my notebook!<br>
#     I will introduce the steps for data science in this notebook.<br>
#     This is the overview of the notebook below.<br>
# </div>  
# 
# | # | Contents | Discription |
# | --- | --- | --- |
# | 1 | Import Library | Install and import libraries you use. |
# | 2 | Read Data | Show how to read the train and test data. |
# | 3 | EDA | Get the insight of data. I introduce Simple way. |
# | 4 | Feature Engineering | Transform data by Simple way. |
# | 5 | Build Some Models | Build and train some models like Sklearn Models, GBDT, NN, Ensenmble. | 
# | 6 | Submit Your Result | Show how to submit your result to leaderboard. |

# <H1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Import Library</b>
# </H1>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     At first, let's install and import library. Some libraries cannot be used by default. You can install 3rd-party libraries as below.
# </div>  

# In[1]:


get_ipython().system('pip install -q pandas-profiling')
get_ipython().system('pip install -q pytorch-tabnet')
get_ipython().system('pip install -q deeptables')
from IPython.display import clear_output
clear_output()


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     '-q' or '--quiet' gives you less output. You can skip long annoying install outputs.
# </div>  

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Next, Let's imort libraries.
# </div>  

# In[2]:


import warnings
warnings.filterwarnings("ignore") # Ignore all warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# for plot result
import matplotlib.pyplot as plt
import seaborn as sns
# for automatic EDA
from pandas_profiling import ProfileReport
# preprocess, split data, model, and metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
# catboost
from catboost import CatBoostClassifier
# XGBoost
from xgboost import XGBClassifier
# lightgbm
from lightgbm import LGBMClassifier
# tensorflow 
import tensorflow as tf
# pytorch
import torch
# tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
# deeptabels
from deeptables.models import deeptable, deepnets

# Check the path of data
import os
for dirname, _, filenames in os.walk('/kaggle/input/titanic'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
SEED = 42 # for Reproducibililty


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Above outputs mean the paths of input data that you read.
# </div>  
# 
# | File Path | Description | 
# | --- | --- |
# |/kaggle/input/titanic/train.csv | Training data path. | 
# | /kaggle/input/titanic/test.csv | Test data path. |
# | /kaggle/input/titanic/gender_submission.csv | The template of submission csv. |

# <H1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Read Data</b>
# </H1>
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Next, Let's read the train and test data you see above.
# </div>

# In[3]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Check the data.
# </div>

# In[4]:


train_df.info()


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Let's see the output briefly.<br>
#     <b>RangeIndex: 891 entries, 0 to 890</b> means the training data have 891 records.The index is 0 to 890.<br>
#     <b>Data columns (total 12 columns)</b> means the columns are 12.<br>
#     <b>Non-Null Count</b> means the number of not null values. If the number is less than 891, there are some null values.<br>
#     <b>Dtype</b> is the column's data type. 'object' almost means string.<br>
#     <br>
#     The means of each column is <a ref="https://www.kaggle.com/competitions/titanic/data" style="border-bottom:solid; border-color: #808080; border-width: 2px;">here</a>
# </div>

# <h1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>EDA</b>
# </h1>
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Next, to gain insights of the data, we try EDA(Exploratory Data Analysis).
# </div>
# 
# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Adversarial Validation</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Before we explore data itself, we try 'Adversarial Validation'.<br>
#     We can find the difference between train and test data. If the train and test data are much different, we must resample data and reduce the difference. It is important to improve the competitions score. Let's do it!
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     I concatenate train and test data. And I drop the 'Survived' colummn because the test data doesn't have the column. Next, I add 'is_test' column. For adversarial validation, 'is_test' is the target column and we try to predict 'is_test' value. If we can predict correctly, this means train and test data have different distribution. If we cannot predict, this means there is no difference between train and test data. I drop the 'Name', 'Ticket' and 'Cabin' columns because these will be different between train and test data. This is the unique values.
# </div>

# In[5]:


all_df = pd.concat([train_df, test_df], axis=0)
all_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
all_df['is_test'] = [0]*train_df.shape[0] + [1]*test_df.shape[0]
all_df.head()


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Set 'feature_cols' and 'target_col'. feature_cols are the column for training data of adversarial validation. target_col is the target column for predicting whether the record is test data or not.
# </div>

# In[6]:


feature_cols = all_df.columns.tolist()
feature_cols.remove('is_test')
target_col = 'is_test'


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     set X is train data for adversarial validation and y is test data.
# </div>

# In[7]:


X = all_df[feature_cols]
y = all_df[target_col]


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     To treat string type data, we use label encoder. It transform string to integer.
# </div>

# In[8]:


le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X['Embarked'] = le.fit_transform(X['Embarked'])


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Finally, let's predict 'is_test'!
# </div>

# In[9]:


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=SEED)


# In[10]:


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
model = LGBMClassifier(random_state=SEED)
scores = cross_validate(model, train_X, train_y, scoring='roc_auc', cv=kf, n_jobs=-1, return_estimator=True)


# In[11]:


y_pred = np.mean([model.predict_proba(val_X)[:, 1] for model in scores['estimator'] ], axis=0)


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Check ROC AUC metric. It is effective metric for binary classification. In accurate prediction, the value is close to 1. In bad prediction such as random prediction, the value is close to 0.5.
# </div>

# In[12]:


roc_auc_score(y_true=val_y, y_score=y_pred)


# In[13]:


fpr, tpr, thresholds = roc_curve(val_y, y_pred)
plt.plot(fpr, tpr)
plt.title('ROC Curve', fontsize=20)
plt.xlabel('FPR: False Positive Rate', fontsize=15)
plt.ylabel('TPR: True Positive Rate', fontsize=15)
plt.show()


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     As you see above, roc_auc_score is almost 0.5 and ROC Curve is almost linear. This means that the model cannot distinguish train and test data. So we do not need additional treatment for train data. If roc_auc_score is larger than 0.6~0.8, some train data do not represent test data so we need to resample or do some transformation.
# </div>

# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Auto EDA</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In this section, I introduce very useful tool for EDA, <b>pandas-profiling</b>. We can easily plot and get insights by pandas-profiling automatically. Let's use it!
# </div>

# In[14]:


profile = ProfileReport(train_df, title='Titanic data profile')


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     If you copy the notebook, please uncomment the below cell and try running! 
# </div>

# In[15]:


# profile


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Look at <b>Overview > Overview</b>. We can check the number of columns, rows, missing values.<br>
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     We can know the feature of each columns in <b>Overview > Alert</b>. We can find how high cadinality, many missing values and so on.<br>
# </div>

# ![image2.png](attachment:66bdd578-507a-4ee1-974b-ae38d1509f0a.png)

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In <b>Variables</b>, we can check the values and histgrams of each columns.<br>
# </div>

# ![image3.png](attachment:bc6b81e5-0504-4f89-a6eb-b6706db638aa.png)

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In <b>Interaction</b>, we can check the scatter plot of two columns.<br>
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In <b>Correlation</b>, we can check the correlation heatmap.<br>
# </div>

# ![image5.png](attachment:261c3675-9b3e-45e1-b92b-8345660d77b4.png)

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In <b>Missing Values</b>, we can check the how many missing values is in each columns visually.<br>
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Finally in <b>Sample</b>, we can check first 10 columns and last 10 columns.
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In Auto EDA, we can know the things below.
#     <ol>
#         <li>'Age', 'Cabin' and 'Embarked' have missing values, especially 'Cabin' have so many.</li>
#         <li>'Passenge_id', 'Ticket', 'Cabin' have many unique values.</li>
#         <li>'Pclass', 'Age' and 'Sex' have slightly high correlation for 'Survived'.</li>
#     </ol>  
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     To get high prediction accuracy, I come back to EDA and explore more details. 
# </div>

# <h1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Feature Engineering</b>
# </h1>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In this section, I simply try the process below.
#     <ol>
#         <li>Fill missing values for 'Age', 'Embarked'.</li>
#         <li>Standardize 'Age', 'Fare'.</li>
#         <li>Get the number of passenger of each group.</li>
#         <li>Drop the unique and high cardinal columns, like 'PassengerId', 'Name', 'Ticket' and 'Cabin'.</li>
#     </ol>
#     Let's define the function for preprocessing.
# </div>

# In[16]:


def titanic_preprocessing(input_df):
    df = input_df.copy()
    # missin values
    # Age is int(or float) so use mean or median
    # Embarked is caterogy(label) so use mode
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median()) # Fare is missing in test data
    # extract title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # Label Encoding
    # let the string category feature to integer
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title'] = le.fit_transform(df['Title'])
    # Standardize
    # Scaling int or float feature, it improve prediction for especially Neural Network Model.
    sc = StandardScaler()
    df[['Age', 'Fare']] = sc.fit_transform(df[['Age', 'Fare']])
    # Get the number of passenger in each group
    df['PassengersInGroup'] = df['SibSp'] + df['Parch'] + 1 # Siblings/Spouses + Parent/Children + him/herself
    df['IsAlone'] = df['PassengersInGroup'].apply(lambda x: 1 if x == 1 else 0)
    # drop sibsp and parch because these have high correlation for PassengersInGroup
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
    return df


# In[17]:


train_data = titanic_preprocessing(train_df)
train_data.head()


# <H1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Build and Fit Model</b>
# </H1>
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     In this section, we build and fit some models, like RandomForest, XGBoost, LightGBM, Keras, Pytorch.<br>
#     I set the least hyperparamter, so if you want to know the hyperparameter more, check the other my notebook, introduced soon.
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Before training, I define the reset seed function. It is useful for reproducibility. I describe more at <a ref="https://github.com/ydataai/ydata-profiling" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Discussion: Recommendation to set your seed!.</a><br><br>
# </div>

# In[18]:


def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # for CUDA
    torch.backends.cudnn.deterministic = True # for CUDNN
    torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest


# In[19]:


# target column
target_col = 'Survived'
# columns, using for prediction
feature_cols = train_data.columns.tolist()
feature_cols.remove(target_col)


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Logistic Regression</b>
# </h2>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Logistic Regression is the simple classification model that works by fitting a logistic function.
# </div>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     To aviod overfitting, I use KFold and Cross Validation. Stratified K-Fold balances target values(Survived = 0 and 1) ratio in train and validation data.
# </div>

# In[20]:


X, y = train_data[feature_cols], train_data[target_col]
 # split data to 'n_split' data and set each data to validation data. other data are used for training
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)


# In[21]:


lr_model = LogisticRegression(random_state=SEED)
scores = cross_validate(lr_model, X, y, scoring='accuracy', cv=skf, n_jobs=-1, return_estimator=True)


# In[22]:


lr_models = scores['estimator']


# In[23]:


y_scores = [ model.predict(X) for model in lr_models]
accuracy = np.mean([accuracy_score(y_score, y) for y_score in y_scores])
print(f'Logistic Regression Accuracy(All data) = {accuracy:.4f}')
print(f'Logistic Regression Accuracy(Mean score) = {np.mean(scores["test_score"]):.4f}')


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Support Vector Machine</b>
# </h2>

# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#      Support Vector Machine works by finding the hyperplane that best separates the data into different classes.<br>
#     Also I use KFold and Cross Validation.
# </div>

# In[24]:


X, y = train_data[feature_cols], train_data[target_col]
 # split data to 'n_split' data and set each data to validation data. other data are used for training
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)


# In[25]:


svc_model = SVC(
    probability=True, 
    random_state=SEED)
scores = cross_validate(svc_model, X, y, scoring='accuracy', cv=skf, return_estimator=True)


# In[26]:


svc_models = scores['estimator']


# In[27]:


y_scores = [ model.predict(X) for model in svc_models]
accuracy = np.mean([accuracy_score(y_score, y) for y_score in y_scores])
print(f'Support Vector Machine Accuracy(All data) = {accuracy:.4f}')
print(f'Support Vector Machine Accuracy(Mean score) = {np.mean(scores["test_score"]):.4f}')


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>RandomForest</b>
# </h2>

# In[28]:


X, y = train_data[feature_cols], train_data[target_col]
 # split data to 'n_split' data and set each data to validation data. other data are used for training
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)


# In[29]:


rf_model = RandomForestClassifier(random_state=SEED)
scores = cross_validate(rf_model, X, y, scoring='accuracy', cv=skf, return_estimator=True)


# In[30]:


rf_models = scores['estimator']


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Check the accuracy for all training data.
# </div>

# In[31]:


y_scores = [ model.predict(X) for model in rf_models]
accuracy = np.mean([accuracy_score(y_score, y) for y_score in y_scores])
print(f'Random Forest Accuracy(All data) = {accuracy:.4f}')
print(f'Random Forest Accuracy(Mean score) = {np.mean(scores["test_score"]):.4f}')


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Gradient Boosted Decision Tree(GBDT)</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     GBDT can treat null values themselves. To see this, we modify preprocessing.
# </div>

# In[32]:


def titanic_preprocessing_for_gbdt(input_df):
    df = input_df.copy()
    # missin values
    # Age is int(or float) so use mean or median
    # Embarked is caterogy(label) so use mode
    # df['Age'] = df['Age'].fillna(df['Age'].median())
    # df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # df['Fare'] = df['Fare'].fillna(df['Fare'].median()) # Fare is missing in test data
    # extract title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # Label Encoding
    # let the string category feature to integer
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title'] = le.fit_transform(df['Title'])
    # Standardize
    # Scaling int or float feature, it improve prediction for especially Neural Network Model.
    sc = StandardScaler()
    df[['Age', 'Fare']] = sc.fit_transform(df[['Age', 'Fare']])
    # Get the number of passenger in each group
    df['PassengersInGroup'] = df['SibSp'] + df['Parch'] + 1 # Siblings/Spouses + Parent/Children + him/herself
    df['IsAlone'] = df['PassengersInGroup'].apply(lambda x: 1 if x == 1 else 0)
    # drop sibsp and parch because these have high correlation for PassengersInGroup
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
    return df


# In[33]:


train_data_for_gbdt = titanic_preprocessing_for_gbdt(train_df)
train_data_for_gbdt.info()


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>CatBoost</b>
# </h2>

# In[34]:


X, y = train_data_for_gbdt[feature_cols], train_data_for_gbdt[target_col]
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)


# In[35]:


cb_model = CatBoostClassifier(random_seed=SEED)
scores = cross_validate(cb_model, X, y, scoring='accuracy', cv=skf, return_estimator=True)
# output is long so clear after training
clear_output()


# In[36]:


cb_models = scores['estimator']


# In[37]:


y_scores = [ model.predict(X) for model in cb_models]
accuracy = np.mean([accuracy_score(y_score, y) for y_score in y_scores])
print(f'CatBoost Accuracy(All data) = {accuracy:.4f}')
print(f'CatBoost Mean Score = {np.mean(scores["test_score"]):.4f}')


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>XGBoost</b>
# </h2>

# In[38]:


X, y = train_data_for_gbdt[feature_cols], train_data_for_gbdt[target_col]
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)


# In[39]:


xgb_model = XGBClassifier(random_state=SEED)
scores = cross_validate(xgb_model, X, y, scoring='accuracy', cv=skf, return_estimator=True)


# In[40]:


xgb_models = scores['estimator']


# In[41]:


y_scores = [ model.predict(X) for model in xgb_models]
accuracy = np.mean([accuracy_score(y_score, y) for y_score in y_scores])
print(f'XGBoost Accuracy(All data) = {accuracy:.4f}')
print(f'XGBoost Accuracy(Mean score) = {np.mean(scores["test_score"]):.4f}')


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>LightGBM</b>
# </h2>

# In[42]:


X, y = train_data_for_gbdt[feature_cols], train_data_for_gbdt[target_col]
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)


# In[43]:


lgb_model = LGBMClassifier(random_state=SEED)
scores = cross_validate(lgb_model, X, y, scoring='accuracy', cv=skf, return_estimator=True)


# In[44]:


lgb_models = scores['estimator']


# In[45]:


y_scores = [ model.predict(X) for model in lgb_models]
accuracy = np.mean([accuracy_score(y_score, y) for y_score in y_scores])
print(f'LightGBM Accuracy(All data) = {accuracy:.4f}')
print(f'LightGBM Accuracy(Mean score) = {np.mean(scores["test_score"]):.4f}')


# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Neural Network(NN) Model</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     For tabular data, GBDT is so strong models but some Deep NN models is as good as GBDT.<br>
#     In this section I introduce two NN models, TabNet and DeepTables.
# </div>

# <h2 style="font-size: 20pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>TabNet</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     The first model is <a ref="https://arxiv.org/pdf/1908.07442.pdf" style="border-bottom: solid; border-color: #808080; border-width: 2px;">TabNet</a>. Thanks to <a ref="https://github.com/dreamquark-ai/tabnet" style="border-bottom: solid; border-color: #808080; border-width: 2px;">pytorch-tabnet</a>, we can easily use TabNet.<br>
#     In NN models, I use train_test_split for simplicity.
# </div>

# In[46]:


X, y = train_data[feature_cols], train_data[target_col]
train_X, val_X, trai_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)


# In[47]:


tabnet_model = TabNetClassifier(
    n_d=64, n_a=64, n_steps=10,  
    optimizer_fn=torch.optim.Adam, 
    optimizer_params=dict(lr=1e-3),
    seed=SEED
)
tabnet_model.fit(
    train_X.values, train_y.values,
    max_epochs=20,
    batch_size=64,
    patience=0,
    eval_set=[(val_X.values, val_y.values)],
    eval_metric=['accuracy']
)
clear_output()


# In[48]:


y_score_all = tabnet_model.predict(X.values)
y_score_val = tabnet_model.predict(val_X.values)
print(f'TabNet Accuracy(All data) = {accuracy_score(y_score_all, y):.4f}')
print(f'TabNet Accuracy(Validation data) = {accuracy_score(y_score_val, val_y):.4f}')


# <h2 style="font-size: 20pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>DeepTables</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     The second model is <a ref="https://github.com/dreamquark-ai/tabnet" style="border-bottom: solid; border-color: #808080; border-width: 2px;">deeptables</a>.
# </div>

# In[49]:


X, y = train_data[feature_cols], train_data[target_col]
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)


# In[50]:


dt_config = deeptable.ModelConfig(nets=deepnets.DeepFM, metrics=['accuracy'])
dt_model = deeptable.DeepTable(config=dt_config)
model, history = dt_model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=64,
    epochs=10
)
clear_output()


# In[51]:


y_score_all = dt_model.predict(X)
y_score_val = dt_model.predict(val_X)
clear_output()
print(f'DeepTable Accuracy(All data) = {accuracy_score(y_score_all, y):.4f}')
print(f'DeepTable Accuracy(Validation data) = {accuracy_score(y_score_val, val_y):.4f}')


# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     My TabNet and DeepTable model are bit worse than GBDT but this may improve hyperparameter tuning.
# </div>

# <h2 style="font-size: 25pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Ensemble</b>
# </h2>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     We build some models. Next, I ensemble them. I introduce some ensemble methods in the notebook: <a ref="https://www.kaggle.com/code/masatakasuzuki/4-ensemble-methods" style="border-bottom:solid; border-color: #808080; border-width: 2px;">🚀4 Ensemble Methods</a>
# </div>

# In[52]:


X, y = train_data[feature_cols], train_data[target_col]
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)


# In[53]:


lr_params = {'C': 0.046415888336127774}
svc_params = {'C': 0.36870095353565224, 'gamma': 'scale', 'kernel': 'linear'}
rf_params = {
    'max_depth': 7,
    'min_samples_leaf': 1,
    'min_samples_split': 6,
    'n_estimators': 70,
}
gb_params = {
    'learning_rate': 0.009679572227018537,
    'max_depth': 5,
    'min_samples_leaf': 6,
    'min_samples_split': 6,
    'n_estimators': 267
}
hgb_params = {
    'l2_regularization': 15.601864044243651,
    'learning_rate': 0.020176745817583806,
    'max_bins': 106,
    'max_iter': 5588,
    'max_leaf_nodes': 89,
    'min_samples_leaf': 101
}
xgb_params = {
    'n_estimators': 1618,
    'learning_rate': 0.015010643054191333,
    'min_child_weight': 3,
    'max_depth': 47,
    'max_delta_step': 12,
    'subsample': 0.32078976980620055,
    'colsample_bytree': 0.7038655569988447,
    'colsample_bylevel': 0.9110924308029994,
    'reg_lambda': 3.4823803155558456e-06,
    'reg_alpha': 0.02927434615116004,
    'gamma': 0.0036972686999563963,
    'scale_pos_weight': 1.202503053965045
}
cb_params = {
    'iterations': 222,
    'depth': 2,
    'learning_rate': 0.364311438409535,
    'random_strength': 0.05538761955351006,
    'bagging_temperature': 0.04630917740322642,
    'border_count': 161,
    'l2_leaf_reg': 28,
    'scale_pos_weight': 1.1660605712254726
}
lgb_params = {
    'n_estimators': 398,
    'learning_rate': 0.03548719621444271,
    'colsample_bytree': 0.6813507676306729,
    'num_leaves': 34107,
    'subsample': 0.6578823118178739,
    'reg_lambda': 2.1567236451010223,
    'reg_alpha': 1.7109456653529398e-05,
    'min_child_samples': 51
}

lr_model= LogisticRegression(**lr_params, random_state=SEED)
kn_model = KNeighborsClassifier()
svc_model = SVC(**svc_params, probability=True, random_state=SEED)
rf_model = RandomForestClassifier(**rf_params, random_state=SEED)
gb_model = GradientBoostingClassifier(**gb_params, random_state=SEED)
hgb_model = HistGradientBoostingClassifier(**hgb_params, random_state=SEED)
cb_model = CatBoostClassifier(**cb_params, random_seed=SEED)
xgb_model = XGBClassifier(**xgb_params, random_state=SEED)
lgb_model = LGBMClassifier(**lgb_params, random_state=SEED)
meta_estimator = LogisticRegression(random_state=0)

stacking_classifier = StackingClassifier(
    estimators=[
        ('logistic regression', lr_model),
        ('k neighbors', kn_model),
        ('svc', svc_model),
        ('random forest', rf_model),
        ('gradient boosting', gb_model),
        ('hist gradient boosting', hgb_model),
        ('catboost', cb_model),
        ('xgboost', xgb_model),
        ('lightgbm', lgb_model),
    ],
    final_estimator=meta_estimator,
    cv=10
)
stacking_classifier.fit(train_X, train_y)
clear_output()


# In[54]:


y_score_all = stacking_classifier.predict(X)
y_score_val = stacking_classifier.predict(val_X)
print(f'Ensemble Accuracy(All data) = {accuracy_score(y_score_all, y):.4f}')
print(f'Ensemble Accuracy(Validation data) = {accuracy_score(y_score_val, val_y):.4f}')


# <h1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Predict and Submit Your Result</b>
# </h1>
# 
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Finally, predict and submit your result. I use Stacking Classiflier.
# </div>

# In[55]:


test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data = titanic_preprocessing(test_df)
test_X = test_data[feature_cols]
test_X.head()


# In[56]:


submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
y_pred = stacking_classifier.predict(test_X)
submission_df[target_col] = y_pred
submission_df.to_csv('submission.csv', index=False)


# <div style="font-size: 20pt; font-family: 'Source Sans Pro', sans-serif;">
#     <b>
#         Thank you for reading this notebook. Please give me any feedback!
#     </b>
# </div>

# <H1 style="font-size: 35pt; color: #58B2DC; font-family: 'Source Sans Pro', sans-serif;">
#     <b>Reference</b>
# </H1>
# <div style="font-size: 15pt; font-family: 'Source Sans Pro', sans-serif;">
#     Title Ship Photo: <a ref="https://unsplash.com/ja/%E5%86%99%E7%9C%9F/ToRz-jwncrM" style="border-bottom:solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     Adversarial validation: Medium is helpful, for example <a ref="https://medium.com/towards-data-science/how-to-assess-similarity-between-two-datasets-adversarial-validation-246710eba387" style="border-bottom: solid; border-color: #808080; border-width: 2px;">this link.</a><br>
#     pandas-profiling: <a ref="https://github.com/ydataai/ydata-profiling" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     Recommendation to set your seed!: <a ref="https://www.kaggle.com/discussions/getting-started/395015" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     Sklearn RandomForest: <a ref="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     CatBoost: <a ref="https://catboost.ai/" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     XGBoost: <a ref="https://xgboost.readthedocs.io/en/stable/" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     LightGBM: <a ref="https://lightgbm.readthedocs.io/" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Link</a><br>
#     TabNet: <a ref="https://arxiv.org/pdf/1908.07442.pdf" style="border-bottom: solid; border-color: #808080; border-width: 2px;">Arxiv Link</a>, <a ref="https://github.com/dreamquark-ai/tabnet" style="border-bottom: solid; border-color: #808080; border-width: 2px;">pytorch-tabnet</a><br>
#     DeepTables: <a ref="https://github.com/dreamquark-ai/tabnet" style="border-bottom: solid; border-color: #808080; border-width: 2px;">deeptables</a><br>
# </div>
