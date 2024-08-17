#!/usr/bin/env python
# coding: utf-8

# ## **Notebook Imports**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import tensorflow as tf

import optuna


# ## **Importing Data**

# In[2]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()


# In[3]:


test.head()


# In[4]:


target = train['Survived'].astype(int)
test_ids = test['PassengerId']


# In[5]:


train1 = train.drop(['PassengerId', 'Survived'], axis= 1)
test1 = test.drop('PassengerId', axis=1)


# In[6]:


data1 = pd.concat([train1, test1], axis= 0).reset_index(drop= True)
data1.head()


# In[7]:


data1.isna().sum()


# In[8]:


data1.drop(['Name', 'Ticket', 'Cabin'], axis= 1, inplace= True)


# In[9]:


data1.head()


# ## **EDA**

# In[10]:


sns.countplot(x = target, palette= 'winter')
plt.xlabel('Titanic Survival Rate');


# So, it's clear from the above plot that majority of the people onboarding the titanic did not survived.

# In[11]:


plt.figure(figsize= (16, 8))
sns.heatmap(data1.corr(), annot = True, cmap= 'YlGnBu', fmt= '.2f');


# It seems like most of our independent varaibles are not correlated except `SibSp` and `Parch`. We will deal with that while doing feature engineering

# In[12]:


sns.set_context('notebook', font_scale= 1.2)
fig, ax = plt.subplots(2, figsize = (20, 13))

plt.suptitle('Distribution of Age and Fair based on target variable', fontsize = 20)

# I am using the training dataset only to plot these as we don't have target variable in our test dataset
ax1 = sns.histplot(x ='Age', data= train, hue= 'Survived', kde= True, ax= ax[0], palette= 'winter')
ax1.set(xlabel = 'Age', title= 'Distribution of Age based on target variable')

ax2 = sns.histplot(x ='Fare', data= train, hue= 'Survived', kde= True, ax= ax[1], palette= 'viridis')
ax2.set(xlabel = 'Fare', title= 'Distribution of Fare based on target variable')

plt.show()


# It is evident from the plot that children did tend to have more chances of survival as compared to older individuals
# 

# In[13]:


sns.countplot(x = 'Sex', data= train, hue= 'Survived', palette= 'pastel')
plt.title('Survival chance based on Gender', fontsize = 15);


# Now that's a clear pattern here. It seems like females were 3 times more likely to survive as compared to males.

# In[14]:


sns.countplot(x = 'Pclass', data= train, hue= 'Survived', palette= 'pastel')
plt.title('Survival chance based on Ticket Class', fontsize = 15);


# We can also conclude that people travelling in 3rd class were less likely to survive as compared to people travelling in first class
# 

# ## **Filling Missing Values**

# In[15]:


def knn_impute(df, na_target):
    df = df.copy()
    
    numeric_df = df.select_dtypes(np.number)
    non_na_columns = numeric_df.loc[: ,numeric_df.isna().sum() == 0].columns
    
    y_train = numeric_df.loc[numeric_df[na_target].isna() == False, na_target]
    X_train = numeric_df.loc[numeric_df[na_target].isna() == False, non_na_columns]
    X_test = numeric_df.loc[numeric_df[na_target].isna() == True, non_na_columns]
    
    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    df.loc[df[na_target].isna() == True, na_target] = y_pred
    
    return df


# In[16]:


sns.histplot(data1['Age'], color= 'teal', kde= True);


# In[17]:


data2 = knn_impute(data1, 'Age')


# In[18]:


sns.countplot(x = data2['Embarked'], palette= 'Set2');


# Majority of the people embarked from Southampton, so we will just fill the missing values in Embarked column with `S`

# In[19]:


data2['Embarked'].fillna('S', inplace= True)


# In[20]:


plt.figure(figsize= (10, 6))
sns.histplot(data2['Fare'], color= 'Teal', kde= True);


# Distribution of `Fare` is clearly skewed, therefore we will just fill the only missing value we have in this column with median

# In[21]:


data2['Fare'].fillna(data1['Fare'].median(), inplace= True)


# In[22]:


data2.isna().sum()


# We don't have any more missing values in the dataset. Let's now move on to Encoding our categorical variables

# ## **Encoding**

# In[23]:


data3 = data2.copy()
data3.info()


# In[24]:


data3 = pd.get_dummies(data3)


# In[25]:


data3.head()


# ## **Scaling**

# In[26]:


sc = StandardScaler()
data3[['Age', 'Fare']] = sc.fit_transform(data3[['Age', 'Fare']])


# In[27]:


data3.head()


# ## **Feature Enginerring**

# Both the `SibSp` and `Parch` column suggests whether the person was person was travelling with his family or not. So we will convert these features into a single feature called family

# In[28]:


data3['Family'] = np.where(data3['SibSp'] + data3['Parch'] > 0, 1, 0)
data3.drop(['SibSp', 'Parch'], axis= 1, inplace= True)


# In[29]:


data3.head()


# In[30]:


train_final = data3.loc[:train.index.max(), :].copy()
test_final = data3.loc[train.index.max() + 1:, :].reset_index(drop=True).copy()


# ## **Modelling**

# In[31]:


models = {
    'xgboost' : XGBClassifier(),
    'catboost' : CatBoostClassifier(verbose=0),
    'lightgbm' : LGBMClassifier(),
    'gradient boosing' : GradientBoostingClassifier(),
    'random forest' : RandomForestClassifier(),
    'logistic regression': LogisticRegression(),
    'naive bayes': GaussianNB(),
}


# In[32]:


target = target.astype(int)


# In[33]:


for name, model in models.items():
    model.fit(train_final, target)
    print(f'{name} trained')


# ## **Evaluation**

# In[34]:


results = {}

kf = KFold(n_splits= 10)

for name, model in models.items():
    result = cross_val_score(model, train_final, target, scoring = 'roc_auc', cv= kf)
    results[name] = np.mean(result)


# In[35]:


for name, result in results.items():
    print("-------\n" + name)
    print(f'ROC score: {round(result, 3)}')


# In[36]:


results_df = pd.DataFrame(results, index=range(0,1)).T.rename(columns={0: 'ROC Score'}).sort_values('ROC Score', ascending=False)
results_df


# In[37]:


plt.figure(figsize = (15, 6))
sns.barplot(x= results_df.index, y = results_df['ROC Score'], palette = 'summer')
plt.xlabel('Model')
plt.ylabel('ROC Score')
plt.title('ROC Score of different models');


# #### Clearly, catboost had the best ROC Score, lets do some hyperparameter optimization and make the final predictions based on it

# ## **Hyperparamer Optimization**

# In[38]:


# def catboost_objective(trial):
#     learning_rate = trial.suggest_float('learning_rate', 0, 0.5)
#     depth = trial.suggest_int('depth', 3, 10)
#     n_estimators = trial.suggest_int('n_estimators', 50, 600)
    
#     model = CatBoostClassifier(
#         learning_rate= learning_rate,
#         depth= depth,
#         n_estimators= n_estimators,
#         verbose= 0
#     )

#     model.fit(train_final, target)
#     cv_score = cross_val_score(model, train_final, target, scoring= 'roc_auc', cv= kf)

#     return np.mean(cv_score)

# study = optuna.create_study(direction= 'maximize')
# study.optimize(catboost_objective, n_trials= 100)


# In[39]:


catboost_params = {
    'learning_rate': 0.1682046368673911, 
     'depth': 6, 
     'n_estimators': 540,
     'verbose':0
}


# In[40]:


cb = CatBoostClassifier(**catboost_params)
cb.fit(train_final, target)
pred_cb = cb.predict(test_final)


# ## **ANN**

# In[41]:


train_final.shape


# In[42]:


train_final = np.asarray(train_final).astype(np.float32)
test_final = np.asarray(test_final).astype(np.float32)
target = np.asarray(target).astype(np.float32)


# In[43]:


model = tf.keras.Sequential([
    tf.keras.Input(9),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = 'binary_crossentropy',
    metrics = [tf.keras.metrics.AUC(name='auc')]
)

EPOCHS = 100

history = model.fit(train_final, target, validation_split = 0.20, epochs = EPOCHS)


# In[44]:


plt.figure(figsize=(10, 6))

epochs = range(1, EPOCHS + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# Clearly our ANN is doing a better job in classifying our target column

# In[45]:


pred_ann = model.predict(test_final)


# In[46]:


submission = pd.DataFrame(test_ids, index= None)


# In[47]:


submission['Survived'] = pred_ann


# In[48]:


submission['Survived'] = submission['Survived'].apply(lambda x: 1 if x>0.5 else 0)


# In[49]:


submission.head()


# In[50]:


submission.to_csv('submission.csv', index= None)


# In[ ]:




