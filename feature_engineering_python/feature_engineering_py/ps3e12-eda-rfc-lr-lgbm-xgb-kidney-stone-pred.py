#!/usr/bin/env python
# coding: utf-8

# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Kidney Stone Prediction</p>

# ### <font color='289C4E'>Table of contents<font><a class='anchor' id='top'></a>
# - [Importing Libraries & Exploring the data](#1)
# - [Exploratory Data Analysis](#2)
#     - [Check for Information Bias in Train data and Original Data](#2.1)
#     - [Linear Correlation between the features](#2.2)
#     - [Feature Distributions for train & test data](#2.3)
# - [Feature Engineering](#2.4)
# - [Predictive Analysis](#3)
#     - [Lazypredict : Finding the best perfoming models](#4)
#     - [1. LGBMClassifier](#5)
#     - [2. XGBClassifier](#6)
#     - [3. AdaBoost Classifier](#7)
#     - [4. Random Forest Classifier](#8)
#     - [5. Logistic Regression](#9)
#     - [6. Bonus one: Gaussian Naive Bayes](#10)
# - [Feature Importance](#11)

# ---
# 
# <a id="1"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Importing Libraries & Exploring the data</p>
# 
# ---

# In[1]:


import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from itertools import product


# <h3 align="left"> <font color='blue'>Setting Style</font></h3>

# In[2]:


sns.set_style('darkgrid')


# <h3 align="left"> <font color='blue'>Loading the Data</font></h3>

# In[3]:


train_df = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
original_df = pd.read_csv('/kaggle/input/kidney-stone-prediction-based-on-urine-analysis/kindey stone urine analysis.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s3e12/sample_submission.csv')
train_df


# <h3 align="left"> <font color='blue'>Exploring the Data</font></h3>

# In[4]:


Df = [train_df, original_df, test_df]
names = ['Training Data', 'Original Data','Test Data']
print('Data Information')
for df,name in zip(Df,names):
    print(name)
    print(df.info())
    print()


# In[5]:


train_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)


# In[6]:


desc = train_df.describe()
desc = desc.style.background_gradient()
desc


# ---
# <a id="2"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Exploratory Data Analysis</p>
# ---

# <h2 align="center"><a id="2.1"></a><font color='navy'>Check for Information Bias in Train data and Original Data</font></h2>

# In[7]:


f,ax=plt.subplots(1,2,figsize=(12,10))
train_df['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Target class in training data')
ax[0].set_ylabel('')
original_df['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Target class in original data')
ax[1].set_ylabel('')

plt.show()


# <h3 align="left"> <font color='darkgreen'>As there is almost equal proportion of target class in both train data & original data, <br>Hence there won't be any bias if we merge them<br>
# Threfore let's merge the these two dataframes</font></h3>

# In[8]:


train_df = pd.concat([train_df,original_df],ignore_index=True)
train_df


# <h2 align="center"><a id="2.2"></a><font color='navy'>Linear Correlation between the features</font></h2>

# In[9]:


plt.figure(figsize=(12,8))
sns.heatmap(train_df.corr(),annot=True,cmap='Greens')
plt.title('Correlation Matrix for Features of Train Data');
plt.show()


# <h2 align="center"><a id="2.3"></a><font color='navy'>Feature Distributions for train & test data</font></h2>

# In[10]:


plt.figure(figsize=(10,30))
i = 1
for col in train_df.columns[:6]:
    plt.subplot(6,2,i)
    sns.histplot(x=train_df[col],color='#288BA8',kde=True,lw=1)
    plt.title("training data: distribution of '{}' feature".format(col));
   
    plt.subplot(6,2,i+1)
    sns.histplot(x=test_df[col],color='#B22222',kde=True,lw=1)
    plt.title("testing data: distribution of '{}' feature".format(col));
    i+=2
plt.tight_layout()


# <h2 align="left"> <font color='navy'>Insights</font></h2>
# After going through overall plots we see <br>1. some changes in distribution of <b>urea</b> in train & test data <br>

# In[11]:


plt.figure(figsize=(10,5))
col = 'urea'
plt.subplot(1,2,1)
sns.histplot(x=train_df['urea'],color='#288BA8',kde=True,lw=1)
plt.title("training data: distribution of '{}' feature".format(col));

plt.subplot(1,2,2)
sns.histplot(x=test_df['urea'],color='#B22222',kde=True,lw=1)
plt.title("testing data: distribution of '{}' feature".format(col));
plt.tight_layout()


# The values for <b>urea</b> feature is starting from 64 in test data whereas in train data it is starting from 10 

# In[12]:


train_df[train_df['urea']<50]


# ### So lets drop this values for better predictions

# In[13]:


train_df = train_df[train_df['urea']>50]


# ---
# <a id="2.4"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Feature Engineering</p>
# 
# ---
# 
# The features were added for better performance of the model on the data,
# This feature engineering is taken from [This Awesome Notebook](https://www.kaggle.com/code/phongnguyen1/a-framework-for-tabular-classification-e12-10) by <b>Phong Nguyen</b>. Kudos to this author for his work. You can check that out for more information

# In[14]:


train_df.head(2)


# In[15]:


test_df.head(2)


# In[16]:


def add_features(df):
    # Ratio of calcium concentration to urea concentration: 
    df['calc_urea_ratio'] = df['calc'] / df['urea']
    
#     # Product of calcium concentration and osmolarity: 
#     df['calc_osm_product'] = df['calc'] * df['osmo']
    
#     # Ratio of calcium concentration to specific gravity: 
#     df['calc_gravity_ratio'] = df['calc'] / df['gravity']
    
    # Ratio of calcium concentration to osmolarity: 
    df['calc_osm_ratio'] = df['calc'] / df['osmo']


# In[17]:


train_df


# In[18]:


add_features(train_df)
add_features(test_df)


# In[19]:


train_df.head(3)


# In[20]:


test_df.head(3)


# ---
# <a id="3"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Predictive Analysis</p>
# 
# ---

# <h2 align="center"> <font color='navy'>Standardization for numerical labels</font></h2>

# In[21]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
standardScaler = StandardScaler()


# <h2 align="left"> <font color='navy'>Train Data</font></h2>

# In[22]:


train = standardScaler.fit_transform(train_df.drop(['target'],axis=1))
train = pd.DataFrame(train, columns=train_df.drop(['target'],axis=1).columns)
train


# <h2 align="left"> <font color='navy'>Test Data</font></h2>

# In[23]:


test = standardScaler.fit_transform(test_df)
test = pd.DataFrame(test, columns=test_df.columns)
test


# ---
# <a id="4"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Lazypredict : Finding the best perfoming models</p>
# ---

# In[24]:


get_ipython().system('pip install lazypredict')


# In[25]:


from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split


# In[26]:


X, y = train, train_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# fit all models
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[27]:


models


# <h3 align="left"> <font color='navy'>Above table gives the different models according to their perfomance on the data<br>The models are according to descending ROC AUC which is being use for our model evalution over Test Data<br>
# As we don't know the exact test data, we will try to create all the best models and submit their results into the competition<br> We will use top 4 + logistic regression models for predicting the problem statement , Hence we will be dealing with following baseline models and try for hypertuning for better results<br></font><br>LGBMClassifier<br>XGBClassifier<br>AdaBoostClassifier<br> RandomForestClassifier<br>Logistic Regression</h3>
# 

# ---
# <a id="5"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">1. LGBMClassifier</p>
# ---

# In[28]:


X = train
y = train_df.target
X_test = test


# In[29]:


import lightgbm as lgb
lgbm_params = {'n_estimators': 27, 
               'num_leaves': 5, 
               'min_child_samples': 11, 
               'learning_rate': 0.1,  
               'colsample_bytree': 0.08, 
               'reg_alpha': 1.5,
               'reg_lambda': 0.01
            }
lgb_clf = lgb.LGBMClassifier(**lgbm_params)
# Fitting the model
lgb_clf.fit(X, y)
# Predicting the probabilities of the classes using the model
pred = lgb_clf.predict_proba(X_test)


# In[30]:


# Creting DataFrame of the predicted values
df = pd.DataFrame(pred[:,1])
df.columns = ['target']
df


# In[31]:


# Creating the Data for the submission to competition
sub.drop('target',axis=1,inplace=True)
sub['target']=df['target'].copy()
sub.to_csv('sub_LGBMc.csv', index=False)
sub


# ---
# <a id="6"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">2. XGBClassifier</p>
# ---

# In[32]:


from xgboost import XGBClassifier
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ## Hyperparameter Tuning

# In[33]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# In[34]:


search_space = {
    'n_estimators': [10, 20, 30],
    'max_depth': np.linspace(1, 9, num=5).astype('int'),
    'learning_rate': np.logspace(-3, 1, num=5),
    'reg_alpha': np.linspace(0, 1, num=3),
    'reg_lambda': np.linspace(0, 1, num=3)
}

min_score = 0
best_params = {}
for val in product(*search_space.values()):
    params = {}
    for i, param in enumerate(search_space.keys()):
        params[param] = val[i]
    clf = XGBClassifier(**params).fit(X_train,y_train)
    val_pred=clf.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, val_pred)
    if score > min_score:
        min_score = score
        best_params = params


# In[35]:


best_params


# In[36]:


params = {**best_params,
          'seed':42,
          'eval_metric': 'auc'
         }


# In[37]:


xgb = XGBClassifier(**params)
xgb.fit(X, y)

# Predicting the probabilities of the classes using the model
pred = xgb.predict_proba(X_train)


# In[38]:


# Creting DataFrame of the predicted values
df = pd.DataFrame(pred[:,1])
df.columns = ['target']
df


# In[39]:


# Creating the Data for the submission to competition
sub.drop('target',axis=1,inplace=True)
sub['target']=df['target'].copy()
sub.to_csv('sub_XGBc.csv', index=False)
sub


# ---
# <a id="7"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">3. AdaBoostClassifier</p>
# ---

# In[40]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=400, learning_rate=0.01)
model = abc.fit(X, y)

pred = model.predict_proba(X_test)
pred[:10]


# In[41]:


# Creting DataFrame of the predicted values
df = pd.DataFrame(pred[:,1])
df.columns = ['target']
df


# In[42]:


# Creating the Data for the submission to competition
sub.drop('target',axis=1,inplace=True)
sub['target']=df['target'].copy()
sub.to_csv('sub_AdaBC.csv', index=False)
sub


# ---
# <a id="8"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">4. Random Forest Classifier</p>
# ---

# In[43]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# ## Hyperparameter Tunning

# In[44]:


rfc = RandomForestClassifier()
random_grid = {'bootstrap': [True],
          'max_depth': [25,30],
          'max_features': ['log2','auto'],
          'min_samples_leaf': [2,3,4],
          'min_samples_split': [1,2,3],
          'n_estimators': [170,180]
         }

rf_random = GridSearchCV(estimator = rfc, param_grid = random_grid, cv = 3,n_jobs=-1)
rf_random.fit(X, y)


# In[45]:


rf_random.best_params_


# In[46]:


rf_random.best_params_


# In[47]:


rfc = RandomForestClassifier(**rf_random.best_params_,n_jobs=-1)
rfc.fit(X,y)
pred_rfc = rfc.predict_proba(X_test)
pred_rfc[:5]


# In[48]:


# Creting DataFrame of the predicted values
df_rfc = pd.DataFrame(pred_rfc[:,1])
df_rfc.columns = ['target']
df_rfc


# In[49]:


# Creating the Data for the submission to competition
sub.drop('target',axis=1,inplace=True)
sub['target']=df_rfc['target'].copy()
sub.to_csv('sub_RFc.csv', index=False)
sub


# In[50]:


sub['target'][275]


# ---
# <a id="9"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">5. Logistic Regression</p>
# ---

# In[51]:


from sklearn.linear_model import LogisticRegression


# ## Hyperparameter tunning

# In[52]:


from sklearn.model_selection import GridSearchCV

# Split data into features and target
X = train
y = train_df.target

# Define the logistic regression model
logistic_reg = LogisticRegression()

# Define hyperparameters to tune
hyperparameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.02, 0.05, 0.1],
    'solver': ['liblinear', 'saga','lbfgs'],
    'fit_intercept': [True, False],
    'max_iter': [1, 5, 10, 50, 100],
    'tol': [1e-4, 1e-5]
}

# Perform grid search to find the best hyperparameters
clf = GridSearchCV(logistic_reg, hyperparameters, cv=5)
clf.fit(X, y)

# Print the best hyperparameters and score
print('Best hyperparameters:', clf.best_params_)
print('Best score:', clf.best_score_)


# In[53]:


lr = LogisticRegression(**clf.best_params_)
lr.fit(X,y.values)
pred = lr.predict_proba(X_test)
pred[:5]


# In[54]:


df = pd.DataFrame(pred[:,1])
df.columns = ['target']
df


# In[55]:


sub.drop('target',axis=1,inplace=True)
sub['target']=df['target'].copy()
sub.to_csv('sub_LogR.csv', index=False)
sub


# ---
# <a id="10"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">6. Bonus One : Gaussian Naive Bayes</p>
# ---

# In[56]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


# In[57]:


nb.fit(X, y.values)
pred = nb.predict_proba(X_test)

df = pd.DataFrame(pred[:,1])
df.columns = ['target']

sub.drop('target',axis=1,inplace=True)
sub['target']=df['target'].copy()
sub.to_csv('sub_GaussianNB.csv', index=False)
sub


# In[ ]:





# ---
# <a id="11"></a>
# # <p style="padding:10px;background-color:lightblue;border-style: solid;border-color: black;margin:0;color:green;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 25px 50px;overflow:hidden;font-weight:500">Model Feature Importance</p>
# ---

# **As of now Random forest model done well hence lets take a rfc model for calculating the importance of the features**

# In[58]:


rfc


# In[59]:


df_imp = pd.DataFrame(rfc.feature_names_in_, rfc.feature_importances_)
df_imp.columns = ["Feature_Names"]
df_imp["Importances"] = df_imp.index
df_imp = df_imp.sort_values(by = "Importances", ascending = True)
df_imp.index = np.arange(0,len(df_imp))
df_imp


# In[60]:


plt.figure(figsize = (18,10))
ax = sns.barplot(x = "Feature_Names", y = "Importances", data = df_imp)
plt.title("Feature Importances", fontsize = 20)
for bars in ax.containers:
    ax.bar_label(bars)


# As you can see the added features are playing important role in modelling.
# 
# Hence feature engineering was crucial here, Once again thanks to [Phong Nguyen for this worderful feature engineering](https://www.kaggle.com/code/phongnguyen1/a-framework-for-tabular-classification-e12-10) 

# ---
# # <h2><span style="font-family:Comic Sans MS; color:golden"><strong>If you like it, pls upvote</strong></span></h2>
# <blockquote><h2><span style="color:navy">T</span><span style="color:blue">h</span><span style="color:green">a</span><span style="color:red">n</span><span style="color:red">k</span> <span style="color:green">y</span><span style="color:blue">o</span><span style="color:navy">u</span> 🙂</h2></blockquote>
