#!/usr/bin/env python
# coding: utf-8

# ![](https://i.pinimg.com/originals/29/19/5e/29195e898e78fe7fa5b1da9c5b6d6ba6.gif)

# ## Importing Required Libraries
# <hr style="width:80%;border:1px solid black"> </hr>

# In[1]:


# Mathematical Libraries needed for DataFrame
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
plt.style.use('dark_background')

# Remove unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

# Display DataFrames
from IPython.display import display, display_html

# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

# Splitting Train-Test data
from sklearn.model_selection import train_test_split

# For Cross Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# For showing Progress Bar
from tqdm.notebook import tqdm

# Models
import optuna
import xgboost as xgb
import lightgbm as lgbm
import catboost as catb


# ## Helper Functions
# <hr style="width:80%;border:1px solid black"> </hr>

# In[2]:


def outlier_viz(col):
    '''Distribution and Boxplot for Outlier Detection by @kartik2khandelwal'''
    import random
    color = random.choice(['r', 'g', 'b'])
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    sns.distplot(col, ax=ax[0], color=color)
    sns.boxplot(col, ax=ax[1], color=color)
    plt.suptitle('Distribution & Boxplot for Outlier Detection')
    fig.show()
    return None

def my_print(s):
    '''Custom print function by @kartik2khandelwal :)'''
    a = 4
    for i in s:
        a+=1
    return print('-' * a + '\n' + '| ' + s + ' |' + '\n' + '-' * a)


# ## Loading Dataset
# <hr style="width:80%;border:1px solid black"> </hr>

# In[3]:


df = pd.read_csv('../input/spaceship-titanic/train.csv')
test_data = pd.read_csv('../input/spaceship-titanic/test.csv')
df.head()


# ## Data Field Descriptions
# <hr style="width:80%;border:1px solid black"> </hr>
# 
# * **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.
# * **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * **Destination** - The planet the passenger will be debarking to.
# * **Age** - The age of the passenger.
# * **VIP** - Whether the passenger has paid for special VIP service during the voyage.
# * **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * **Name** - The first and last names of the passenger.
# * **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# ## Exploratory Data Analysis
# <hr style="width:80%;border:1px solid black"> </hr>

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


plt.figure(figsize=(15,5))
sns.heatmap(df.isnull().T, cmap='cool');


# In[8]:


plt.figure(figsize=(12,5))
sns.heatmap(df.corr(), cmap='cool', annot=True, linewidth=0.2, linecolor='black');


# ## Data Visualization
# <hr style="width:80%;border:1px solid black"> </hr>

# #### Target Variable

# In[9]:


plt.figure(figsize=(8,3))
sns.countplot(y = df['Transported'], palette='cool')
plt.title('Checking Class Imbalance');


# In[10]:


cat = [i for i in df.drop(['PassengerId', 'Transported'], axis=1).columns if df[i].nunique() < 10]
num = [i for i in df.drop(['PassengerId', 'Transported'], axis=1).columns if df[i].nunique() > 10]
cat, num


# ### Categorical Data - Visualization

# In[11]:


plt.figure(figsize=(8,5))
sns.countplot(df[cat[0]], palette='cool', hue=df['Transported']); #Most people are from 


# In[12]:


plt.figure(figsize=(8,5))
sns.countplot(df[cat[1]], palette='cool', hue=df['Transported']);


# In[13]:


plt.figure(figsize=(8,5))
sns.countplot(df[cat[2]], palette='cool', hue=df['Transported']);


# In[14]:


plt.figure(figsize=(10,3))
sns.countplot(y=df[cat[3]], palette='cool', hue=df['Transported']);


# ### Numerical Data - Visualization & Outlier Detection

# In[15]:


outlier_viz(df[num[1]])


# In[16]:


outlier_viz(df[num[2]])


# In[17]:


outlier_viz(df[num[3]])


# In[18]:


outlier_viz(df[num[4]])


# In[19]:


outlier_viz(df[num[5]])


# In[20]:


outlier_viz(df[num[6]])


# ### Observations
# > * Cabin - Special feature + needs some cleaning
# > * Age - Almost good disribution with no Outlier
# > * RoomService - Right Skewed with Outliers
# > * FoodCourt - Right Skewed with Outliers
# > * ShoopingMaal - Right Skewed with Outliers
# > * Spa - Right Skewed with Outliers
# > * Name - I don't think so, it will contribute much, but try to find some insights from it.

# ## Feature Engineering
# <hr style="width:80%;border:1px solid black"> </hr>

# #### As per the line - `People in a group are often family members, but not always`. So, taking the surname of each person name might be helpful, because probably the same surname people maybe family members.
# > **Random Thought** - I'm curious to know whether other planet's family members also have same surname ? 🤔🤔
# 
# 
# **Let's extract:**
# - `LastName` from `Name` column, and
# - `Group` from `PassengerId`

# ### Removing Null Values

# In[21]:


df['HomePlanet'].fillna('Earth', inplace=True)
df['CryoSleep'].fillna(False, inplace=True)
df['Destination'].fillna('TRAPPIST-1e', inplace=True)
df['VIP'].fillna(False, inplace=True)
df['Age'].fillna(24.0, inplace=True)


# In[22]:


l = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in l:
    value = df[i].mean()
    df[i] = df[i].fillna(value)


# In[23]:


df.isnull().sum()


# ### Creating and Analysing New Features

# In[24]:


names = df['Name'].str.split(' ', expand=True)
names.columns = ['FirstName', 'LastName']

df = pd.concat([df, names['LastName']],axis=1)
df['LastName'].fillna('None', inplace=True)


# In[25]:


df[['Group', 'GroupNumber']] = df['PassengerId'].str.split('_', expand=True).astype('int32')


# In[26]:


df[['Group', 'LastName', 'Transported']].head()


# In[27]:


temp1 = df[['Group', 'LastName', 'Transported']][2:4].style.set_table_attributes("style='display:inline; margin-right:50px;'").set_caption("Example - 1")
temp2 = df[['Group', 'LastName', 'Transported']][9:12].style.set_table_attributes("style='display:inline; margin-right:50px;'").set_caption("Example - 2")
temp3 = df[['Group', 'LastName', 'Transported']][33:36].style.set_table_attributes("style='display:inline'").set_caption("Example - 3")

display_html(temp1._repr_html_() + temp2._repr_html_() + temp3._repr_html_(), raw=True)


# ### Observation
# > - The trend can clearly be seen, that same group number have same surnames (or family members).
# > - I can also see that, the same family members are either transported or not.<br> Ex:- Consider a family of 2 people with surname 'Ben', it has been observed that if 1 person is transported(or not transported) the other is also transported(or not transported).<br>**PS** - I hope I make my point clear.

# #### Cleaning `Cabin` feature

# In[28]:


df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)
df.drop(['Name', 'Cabin'], axis=1, inplace=True)


# In[29]:


fig, ax = plt.subplots(1,3,figsize=(18,5))
l = ['CabinDeck', 'CabinNum', 'CabinSide']
sns.countplot(df[l[0]], ax=ax[0], palette='cool_r')
sns.distplot(df[l[1]], ax=ax[1], color='blue')
sns.countplot(df[l[2]], ax=ax[2], palette='cool', hue=df['Transported'])
plt.suptitle('Cabin Feature');


# In[30]:


df['AgeGroup'] = pd.cut(df['Age'], bins=20, labels=[i for i in range(1,21)])


# In[31]:


df['CabinDeck'] = df['CabinDeck'].fillna('F')
df['CabinSide'] = df['CabinSide'].fillna('S')
df['CabinNum'] = df['CabinNum'].fillna('82')
df.isnull().sum()


# In[32]:


df['CabinNum'] = df['CabinNum'].astype('int64')
df['AgeGroup'] = df['AgeGroup'].astype('int64')


# In[33]:


df['TotalAmount'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']


# In[34]:


for i in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    df[f'{i}Used'] = df[i].apply(lambda x:1 if x>0 else 0)
df.head()


# In[35]:


df.isnull().sum().sum()


# ## Feature Encoding
# <hr style="width:80%;border:1px solid black"> </hr>

# In[36]:


planet = pd.get_dummies(df['HomePlanet'], drop_first=True)
desti = pd.get_dummies(df['Destination'], drop_first=True)
side = pd.get_dummies(df['CabinSide'], drop_first=True)


# In[37]:


df = pd.concat([df, planet, desti, side], axis=1)
df.drop(['HomePlanet', 'Destination', 'CabinSide'], axis=1, inplace=True)
df.head()


# In[38]:


df['LastName'] = label.fit_transform(df['LastName'])
df['CabinDeck'] = label.fit_transform(df['CabinDeck'])


# In[39]:


df['CryoSleep'] = df['CryoSleep'].map({True:1, False:0})
df['VIP'] = df['VIP'].map({True:1, False:0})
df['Transported'] = df['Transported'].map({True:1, False:0})


# In[40]:


df.head()


# In[41]:


df.to_csv('train_cleaned.csv')


# ## Preprocessing Test Data
# <hr style="width:80%;border:1px solid black"> </hr>

# In[42]:


test_data.head()


# In[43]:


test_data['HomePlanet'].fillna('Earth', inplace=True)
test_data['CryoSleep'].fillna(False, inplace=True)
test_data['Destination'].fillna('TRAPPIST-1e', inplace=True)
test_data['VIP'].fillna(False, inplace=True)
test_data['Age'].fillna(24.0, inplace=True)

l = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in l:
    value = test_data[i].mean()
    test_data[i] = test_data[i].fillna(value)

names = test_data['Name'].str.split(' ', expand=True)
names.columns = ['FirstName', 'LastName']

test_data = pd.concat([test_data, names['LastName']],axis=1)
test_data['LastName'].fillna('None', inplace=True)

test_data[['Group', 'GroupNumber']] = test_data['PassengerId'].str.split('_', expand=True).astype('int32')

test_data[['CabinDeck', 'CabinNum', 'CabinSide']] = test_data['Cabin'].str.split('/', expand=True)
test_data.drop(['Name', 'Cabin'], axis=1, inplace=True)

test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=20, labels=[i for i in range(1,21)])

test_data['CabinDeck'] = test_data['CabinDeck'].fillna('F')
test_data['CabinSide'] = test_data['CabinSide'].fillna('S')
test_data['CabinNum'] = test_data['CabinNum'].fillna('82')

test_data['CabinNum'] = test_data['CabinNum'].astype('int64')
test_data['AgeGroup'] = test_data['AgeGroup'].astype('int64')

test_data['TotalAmount'] = test_data['RoomService'] + test_data['FoodCourt'] + test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck']

for i in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    test_data[f'{i}Used'] = test_data[i].apply(lambda x:1 if x>0 else 0)

planet = pd.get_dummies(test_data['HomePlanet'], drop_first=True)
desti = pd.get_dummies(test_data['Destination'], drop_first=True)
side = pd.get_dummies(test_data['CabinSide'], drop_first=True)

test_data = pd.concat([test_data, planet, desti, side], axis=1)
test_data.drop(['HomePlanet', 'Destination', 'CabinSide'], axis=1, inplace=True)

test_data['LastName'] = label.fit_transform(test_data['LastName'])
test_data['CabinDeck'] = label.fit_transform(test_data['CabinDeck'])

test_data['CryoSleep'] = test_data['CryoSleep'].map({True:1, False:0})
test_data['VIP'] = test_data['VIP'].map({True:1, False:0})


# In[44]:


X_test = test_data.drop('PassengerId', axis=1)
X_test.head()


# In[45]:


test_data.to_csv('test_cleaned.csv')


# ## Splitting Dependent and Independent Variable
# <hr style="width:80%;border:1px solid black"> </hr>

# In[46]:


X = df.drop(['PassengerId', 'Transported'], axis=1)
y = df['Transported']


# ## Splitting Train and Validation data
# <hr style="width:80%;border:1px solid black"> </hr>

# In[47]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=24)


# ## Model Buliding
# <hr style="width:80%;border:1px solid black"> </hr>
# 
# 
# #### NOTE - You can use the below code cells for getting hyperparameters of XGBoost, LGBM & CatBoost using Optuna.

# - <h3>XGBoost

# In[48]:


# import optuna
# import xgboost as xgb
# import lightgbm as lgbm

# dtrain = xgb.DMatrix(X_train, label=y_train)
# dvalid  = xgb.DMatrix(X_valid, label=y_valid)

# def objective(trial):

#     param = {
#         "verbosity": 0,
#         "objective": "multi:softmax",
#         "num_class": 5,
#         'max_depth': trial.suggest_int('max_depth', 2, 15),
#         'n_estimators': 10,
#         # L2 regularization weight.
#         "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
#         # L1 regularization weight.
#         "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
#         # sampling ratio for training data.
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         # sampling according to each tree.
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 2, 20)
#     }

#     bst = xgb.train(param, dtrain)
#     preds = bst.predict(dvalid)
#     pred_labels = np.rint(preds)
#     accuracy = accuracy_score(y_valid, pred_labels)
#     return accuracy


# - <h3> LightGBM

# In[49]:


# import optuna
# import lightgbm as lgbm

# def objective(trial):

#     param = {
#             'metric': 'auc', 
#             'random_state': 24,
#             'n_estimators': 40000,
#             'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt"]),
#             'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
#             'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
#             'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#             'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.6, 0.7, 0.80]),
#             'feature_fraction': trial.suggest_categorical('feature_fraction', [0.6, 0.7, 0.80]),
#             'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3 , 10),
#             'max_depth': trial.suggest_int('max_depth', 2, 12, step=1),
#             'num_leaves' : trial.suggest_int('num_leaves', 13, 148, step=5),
#             'min_child_samples': trial.suggest_int('min_child_samples', 1, 96, step=5),
#         }
    
#     clf = lgbm.LGBMClassifier(**param)  
#     clf.fit(X_train, y_train)

#     preds = clf.predict(X_valid)
#     pred_labels = np.rint(preds)
#     accuracy = accuracy_score(y_valid, pred_labels)
#     return accuracy


# - <h3> CatBoost

# In[50]:


# import optuna
# import catboost as catb

# def objective(trial):

#     param = {
#         "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
#         "depth": trial.suggest_int("depth", 1, 12),
#         "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
#         "bootstrap_type": trial.suggest_categorical(
#             "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
#         ),
#         "used_ram_limit": "3gb",
#     }

#     if param["bootstrap_type"] == "Bayesian":
#         param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
#     elif param["bootstrap_type"] == "Bernoulli":
#         param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    
#     clf = catb.CatBoostClassifier(**param)  
#     clf.fit(X_train, y_train)

#     preds = clf.predict(X_valid)
#     pred_labels = np.rint(preds)
#     accuracy = accuracy_score(y_valid, pred_labels)
#     return accuracy


# In[51]:


# %%time
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100, show_progress_bar=True)


# In[52]:


# new_param = study.best_params
# new_param


# In[53]:


# optuna.visualization.plot_optimization_history(study)


# In[54]:


# optuna.visualization.plot_slice(study)


# ## Model Parameters
# <hr style="width:80%;border:1px solid black"> </hr>

# In[55]:


lgbm_param = {'boosting_type': 'gbdt',
 'lambda_l1': 0.134746148489252,
 'lambda_l2': 0.10521615726990495,
 'colsample_bytree': 0.8,
 'bagging_fraction': 0.7,
 'feature_fraction': 0.6,
 'learning_rate': 0.001019958066572347,
 'max_depth': 9,
 'num_leaves': 23,
 'min_child_samples': 46}

xgb_param = {'max_depth': 6,
 'lambda': 4.5909601357645e-06,
 'alpha': 0.011683603658842521,
 'subsample': 0.6631109810494799,
 'colsample_bytree': 0.3964625968646961,
 'n_estimators': 16,
 'min_child_weight': 11}

catb_param = {'objective': 'CrossEntropy',
 'colsample_bylevel': 0.040629495970693076,
 'depth': 11,
 'boosting_type': 'Plain',
 'bootstrap_type': 'Bernoulli',
 'subsample': 0.7446137182350138}


# ## Model Training
# <hr style="width:80%;border:1px solid black"> </hr>

# In[56]:


N_SPLITS = 50  # previous:300, increasing N_SPLITS to remove error due to randomness

lgbm_preds = []
xgb_preds = []
catb_preds = []

prob = []

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

for fold, (train_id, test_id) in enumerate(tqdm(folds.split(X, y), total=N_SPLITS)):
    X_train = X.iloc[train_id]
    y_train = y.iloc[train_id]
    X_valid = X.iloc[test_id]
    y_valid = y.iloc[test_id]
    
    lgbm_model = lgbm.LGBMClassifier(**lgbm_param)
    xgb_model  = xgb.XGBClassifier(**xgb_param)
    catb_model = catb.CatBoostClassifier(**catb_param, verbose=0)
    
    lgbm_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    catb_model.fit(X_train, y_train)
        
    my_print(f'fold {fold + 1}')
    my_print(f'Training Accuracy   :- {(lgbm_model.score(X_train, y_train)*100).round(2)}% | {(xgb_model.score(X_train, y_train)*100).round(2)}% | {(catb_model.score(X_train, y_train)*100).round(2)}%')
    my_print(f'Validation Accuracy :- {(lgbm_model.score(X_valid, y_valid)*100).round(2)}% | {(xgb_model.score(X_valid, y_valid)*100).round(2)}% | {(catb_model.score(X_valid, y_valid)*100).round(2)}%')
    
    prob1, prob2, prob3 = lgbm_model.predict_proba(X_test), xgb_model.predict_proba(X_test), catb_model.predict_proba(X_test)
    prob.append((prob1 + prob2 + prob3) / 3)
my_print('Model Trained !!!')


# ### <center>☝ Be careful, not to open it ⚠⚠ <center>

# ## Ensembling Models
# <hr style="width:80%;border:1px solid black"> </hr>

# In[57]:


final = [[0,0]]
for i in range(N_SPLITS):
    final = final + prob[i]
    
final = final/N_SPLITS


# In[58]:


y_pred = pd.Series([np.argmax([i]) for i in final])


# In[59]:


from sklearn.metrics import confusion_matrix
xgb_predict = xgb_model.predict(X_valid)
lgbm_predict = lgbm_model.predict(X_valid)
catb_predict = catb_model.predict(X_valid)

fig,ax = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(confusion_matrix(y_valid,  xgb_predict), annot=True, cmap='cool_r', ax=ax[0], square=True)
ax[0].set_title('XGBoost')
sns.heatmap(confusion_matrix(y_valid, lgbm_predict), annot=True, cmap='cool_r', ax=ax[1], square=True)
ax[1].set_title('LGBM')
sns.heatmap(confusion_matrix(y_valid, catb_predict), annot=True, cmap='cool_r', ax=ax[2], square=True)
ax[2].set_title('CatBoost')
plt.suptitle('Confusion Matrix of each Model on Validation Data')
fig.show()


# ## Submission
# <hr style="width:80%;border:1px solid black"> </hr>

# In[60]:


submission = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
submission.head()


# In[61]:


submission['Transported'] = y_pred.astype('bool')
submission.to_csv('submission.csv', index=False)


# <hr style="width:99%;border:1px solid black"> </hr>
# 
# #### <center>Thanks for reading my Notebook, I hope you've learned something from it. 😊</center>
# 
# <hr style="width:99%;border:1px solid black"> </hr>
