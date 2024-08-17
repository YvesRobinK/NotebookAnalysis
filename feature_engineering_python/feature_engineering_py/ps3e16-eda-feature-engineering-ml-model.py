#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf


# In[2]:


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s3e16/train.csv")


# In[3]:


test = pd.read_csv("/kaggle/input/playground-series-s3e16/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s3e16/sample_submission.csv")


# ## Data Preprocessing

# ### Train Data

# In[4]:


train.head(5)


# In[5]:


train.info()


# In[6]:


train.shape


# In[7]:


train.describe()


# In[8]:


train.info()


# ### Test data

# In[9]:


test.head(5)


# In[10]:


test.shape


# In[11]:


test.describe()


# In[12]:


test.info()


# ### EDA : Statistical Analysis

# In[13]:


train.shape


# #### Feature Extraction

# In[14]:


train['Sh/W'] = train['Shucked Weight'] / train['Weight']
test['Sh/W'] = test['Shucked Weight'] / test['Weight']

train['Vi/W'] = train['Viscera Weight'] / train['Weight']
test['Vi/W'] = test['Viscera Weight'] / test['Weight']

train['Sl/W'] = train['Shell Weight'] / train['Weight']
test['Sl/W'] = test['Shell Weight'] / test['Weight']

train['SA'] = train['Length'] + train['Height'] + train['Diameter']
test['SA'] = test['Length'] + test['Height'] + test['Diameter']

train['L/SA'] = train['Length'] / train['SA']
test['L/SA'] = test['Length'] / test['SA']

train['H/SA'] = train['Height'] / train['SA']
test['H/SA'] = test['Height'] / test['SA']

train['D/SA'] = train['Diameter'] / train['SA']
test['D/SA'] = test['Diameter'] / test['SA']


# In[15]:


train.describe()


# In[16]:


test.describe()


# ### Distribution Plots Visualization

# In[17]:


features = train.columns.to_list()


# In[18]:


fig, [top, bottom] = plt.subplots(2, 1, figsize=(8,4))
plt.subplots_adjust(hspace=0)

flierprops = dict(marker='o', markerfacecolor='lightgreen', markeredgecolor='blue', markersize=6)
top.boxplot(x=train['Age'], flierprops=flierprops, vert=False)
top.spines.bottom.set_visible(False)
top.set_xticks([])
top.set_yticks([])

bottom.hist(x=train['Age'], color='lightgreen',edgecolor='blue', bins=29)
bottom.spines.top.set_visible(False)
plt.show()


# In[19]:


features.remove('Age')
features.remove('id')
features.remove('Sex')
cfeatures = ['Sex']


# In[20]:


fig, axes = plt.subplots(7, 2, figsize=(15,25))
plt.subplots_adjust(hspace=0.4)

flierprops = dict(marker='o', markerfacecolor='lightgreen', markeredgecolor='blue', markersize=6)
for ax, numerical_feature in zip(axes.reshape(-1), features):
    ax.boxplot(x=train[numerical_feature], flierprops=flierprops, vert=False)
    ax.set_title(numerical_feature)
    ax.set_yticks([])


# In[21]:


fig, axes = plt.subplots(7, 2, figsize=(15,25))
plt.subplots_adjust(hspace=0.4)

flierprops = dict(marker='o', markerfacecolor='lightgreen', markeredgecolor='blue', markersize=6)
for ax, numerical_feature in zip(axes.reshape(-1), features):
    ax.boxplot(x=test[numerical_feature], flierprops=flierprops, vert=False)
    ax.set_title(numerical_feature)
    ax.set_yticks([])


# In[22]:


train.drop(train[(train['Sh/W']>1) 
                 | (train['Vi/W']>1) 
                 | (train['Sl/W']>1) 
                 | (train['L/SA']<0.4) 
                 | (train['L/SA']>0.6) 
                 | (train['H/SA']>0.25) 
                 | (train['D/SA']<0.3)
                 | (train['D/SA']>0.5)
                ].index, axis=0, inplace=True)

test.loc[test[(test['Sh/W']>1)].index , ['Sh/W', 'Shucked Weight']] = np.nan
test.loc[test[(test['Vi/W']>1)].index , ['Vi/W', 'Viscera Weight']] = np.nan
test.loc[test[(test['Sl/W']>1)].index , ['Sl/W', 'Shell Weight']] = np.nan
test.loc[test[(test['L/SA']>0.6) | ((test['L/SA']<0.4))].index , ['L/SA', 'Length']] = np.nan
test.loc[test[(test['H/SA']>0.25)].index , ['L/SA', 'Height']] = np.nan
test.loc[test[(test['D/SA']>0.5) | (test['D/SA']<0.3)].index , ['L/SA', 'Diameter']] = np.nan


# In[23]:


train.shape


# In[24]:


fig, axes = plt.subplots(7, 2, figsize=(15,25))
plt.subplots_adjust(hspace=0.4)

flierprops = dict(marker='o', markerfacecolor='lightgreen', markeredgecolor='blue', markersize=6)
for ax, numerical_feature in zip(axes.reshape(-1), features):
    ax.boxplot(x=train[numerical_feature], flierprops=flierprops, vert=False)
    ax.set_title(numerical_feature)
    ax.set_yticks([])


# ### Imputing Speed Values from testing set

# In[25]:


imp2 = KNNImputer(n_neighbors=5, missing_values=np.nan)
imp2.fit(train[features])
test[features] = imp2.transform(test[features])


# ### Data Visualization

# In[26]:


sns.histplot(data=train, x='Age', bins=29)
plt.show()


# In[27]:


features


# In[28]:


plt.figure(figsize=(5,5))
sns.countplot(x='Sex', data=train)
plt.show()


# In[29]:


plt.figure(figsize=(15,10))
sns.scatterplot(data = train, x="SA", y="Age", color='lightgreen', edgecolors= 'blue', hue='Sex')
plt.xlabel('SA')
plt.ylabel('Age')
plt.show()


# In[30]:


plt.figure(figsize=(15,10))
sns.scatterplot(data = train, x="L/SA", y="SA", color='lightgreen', edgecolors= 'blue', hue='Age')
plt.xlabel('L/SA')
plt.ylabel('SA')
plt.show()


# In[31]:


plt.figure(figsize=(15,10))
sns.scatterplot(data = train, x="D/SA", y="SA", color='lightgreen', edgecolors= 'blue', hue='Age')
plt.xlabel('D/SA')
plt.ylabel('SA')
plt.show()


# In[32]:


plt.figure(figsize=(15,10))
sns.scatterplot(data = train, x="H/SA", y="SA", color='lightgreen', edgecolors= 'blue', hue='Age')
plt.xlabel('H/SA')
plt.ylabel('SA')
plt.show()


# In[33]:


plt.figure(figsize=(10,8))
sns.scatterplot(data = train, x="Shucked Weight", y="Weight", hue='Sh/W', size='Age',alpha=0.7)
plt.xlabel('Shucked Weight')
plt.ylabel('Weight')
plt.show()


# In[34]:


plt.figure(figsize=(10,8))
sns.scatterplot(data = train, x="Viscera Weight", y="Weight", hue='Vi/W', size='Age',alpha=0.7)
plt.xlabel('Viscera Weight')
plt.ylabel('Weight')
plt.show()


# In[35]:


plt.figure(figsize=(10,8))
sns.scatterplot(data = train, x="Shell Weight", y="Weight", hue='Sl/W', size='Age',alpha=0.7)
plt.xlabel('Shell Weight')
plt.ylabel('Weight')
plt.show()


# ### Encoding Categorial Features

# In[36]:


train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[37]:


train.describe()


# In[38]:


test.describe()


# ## Model Training

# In[39]:


features = train.columns.to_list()
features


# In[40]:


features.remove('id')
features.remove('Age')
X = train[features]
y = train['Age']


# In[41]:


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size =0.2, random_state=1)


# In[42]:


scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)


# In[43]:


fig = plt.figure(figsize=(15,15))
sns.heatmap(data=X.corr(), annot=True, cmap='YlGnBu')
plt.show()


# #### Linear Regression

# In[44]:


from sklearn.linear_model import LinearRegression

pipeline1 = Pipeline([
    ('linearregression', LinearRegression())
])

param_grid = {'linearregression__fit_intercept': [True, False],
              'linearregression__copy_X': [True, False]
              }

grid_search1 = GridSearchCV(pipeline1, param_grid, cv=5)
grid_search1.fit(xtrain, ytrain)

print('Parameters : ', grid_search1.best_params_,'\nAccuracy Score : ', grid_search1.best_score_)


# In[45]:


ypred1 = grid_search1.predict(xtest)
ypred1 = np.round(ypred1)
mae1 = mean_absolute_error(ytest, ypred1)
sc1 = r2_score(ytest, ypred1)
print(f"Validation MAE: {mae1}")
print(sc1)


# #### Decision Tree

# In[46]:


from sklearn.tree import DecisionTreeRegressor

pipeline2 = Pipeline([
    ('clf2', DecisionTreeRegressor(random_state=42))
])

param_grid2 = {
    'clf2__max_depth': [5, 10],
    'clf2__min_samples_split': [2, 5],
    'clf2__min_samples_leaf': [1, 2, 4],
    'clf2__max_leaf_nodes': [None, 5, 10]
}

grid_search2 = GridSearchCV(pipeline2, param_grid2, cv=15)
grid_search2.fit(xtrain, ytrain)

print('Parameters : ', grid_search2.best_params_,'\nAccuracy Score : ', grid_search2.best_score_)


# In[47]:


ypred2 = grid_search2.predict(xtest)
ypred2 = np.round(ypred2)
mae2 = mean_absolute_error(ytest, ypred2)
sc2 = r2_score(ytest, ypred2)
print(f"Validation MAE: {mae2}")
print(sc2)


# In[48]:


best_regressor = grid_search2.best_estimator_

# Obtain the feature importances from the best estimator
importances = best_regressor.named_steps['clf2'].feature_importances_

# Sort the feature importances in descending order

indices = np.argsort(importances)
f=[]
j = list(indices)
for i in j:
    f.append(features[i])

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(xtrain.shape[1]), importances[indices], align="center")
plt.yticks(range(xtrain.shape[1]), f)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# #### Random Forest

# In[49]:


from sklearn.ensemble import RandomForestRegressor
pipeline3 = Pipeline([
    ('clf3', RandomForestRegressor())
])

param_grid3 = {
    'clf3__n_estimators': [200, 500],
    'clf3__max_depth': [10],
}

grid_search3 = GridSearchCV(pipeline3, param_grid3, cv=5)
grid_search3.fit(xtrain ,ytrain)

print('Parameters : ', grid_search3.best_params_,'\nAccuracy Score : ', grid_search3.best_score_)


# In[50]:


ypred3 = grid_search3.predict(xtest)
ypred3 = np.round(ypred3)

mae3 = mean_absolute_error(ytest, ypred3)
sc3 = r2_score(ytest, ypred3)
print(f"Validation MAE: {mae3}")
print(sc3)


# In[51]:


best_regressor = grid_search3.best_estimator_

# Obtain the feature importances from the best estimator
importances = best_regressor.named_steps['clf3'].feature_importances_

# Sort the feature importances in descending order

indices = np.argsort(importances)
f=[]
j = list(indices)
for i in j:
    f.append(features[i])

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(xtrain.shape[1]), importances[indices], align="center")
plt.yticks(range(xtrain.shape[1]), f)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# #### XGBRF

# In[52]:


from xgboost import XGBRFRegressor

pipeline4 = Pipeline([
    ('clf4', XGBRFRegressor())
])

param_grid4 = {
    'clf4__n_estimators': [100, 200],
    'clf4__max_depth': [15],
}

grid_search4 = GridSearchCV(pipeline4, param_grid4, cv=5)
grid_search4.fit(xtrain ,ytrain)

print('Parameters : ', grid_search4.best_params_,'\nAccuracy Score : ', grid_search4.best_score_)


# In[53]:


ypred4 = grid_search4.predict(xtest)
ypred4 = np.round(ypred4)

mae4 = mean_absolute_error(ytest, ypred4)
sc4 = r2_score(ytest, ypred4)
print(f"Validation MAE: {mae4}")
print(sc4)


# In[54]:


best_regressor = grid_search4.best_estimator_

# Obtain the feature importances from the best estimator
importances = best_regressor.named_steps['clf4'].feature_importances_

# Sort the feature importances in descending order

indices = np.argsort(importances)
f=[]
j = list(indices)
for i in j:
    f.append(features[i])

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(xtrain.shape[1]), importances[indices], align="center")
plt.yticks(range(xtrain.shape[1]), f)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# #### ANN

# In[55]:


# Create neural network
ann = tf.keras.models.Sequential()

# create input layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

# create output layer
ann.add(tf.keras.layers.Dense(units=1))

# Compile the network with stochastic gradient descent
ann.compile(optimizer= 'adam', loss= 'mean_squared_error')

# Train the network
ann.fit(xtrain, ytrain, batch_size=32, epochs=100)


# In[56]:


ypred5 = ann.predict(xtest)
ypred5 = np.round(ypred5)
ypred5 = ypred5.reshape(len(ypred5),)

mae5 = mean_absolute_error(ytest, ypred5)
sc5 = r2_score(ytest, ypred5)
print(f"Validation MAE: {mae5}")
print(sc5)


# #### Averaging Ensemble

# In[57]:


ypred6 = np.mean([ypred2, ypred3, ypred4, ypred5], axis=0)

mae6 = mean_absolute_error(ytest, ypred6)
sc6 = r2_score(ytest, ypred6)
print(f"Validation MAE: {mae6}")
print(sc6)


# #### Model Comparison

# In[58]:


f,ax = plt.subplots(1,2, figsize=(20,6))

# Accuracy Score with training and validation scores

ax[0].plot(['Linear\nRegression','Decision\nTree','Random\nForest', 'XG/nBoost', 'ANN'],
         [grid_search1.best_score_,
          grid_search2.best_score_,
          grid_search3.best_score_,
          grid_search4.best_score_,
          sc5],
        color = 'Orange')

ax[0].plot(['Linear\nRegression','Decision\nTree','Random\nForest', 'XG/nBoost', 'ANN', 'Ensmble\nModel'],
         [sc1,sc2,sc3,sc4,sc5,sc6],
        color = 'lightgreen')

ax[0].axhline(y=max([grid_search1.best_score_,
          grid_search2.best_score_,
          grid_search3.best_score_,
          grid_search4.best_score_,
          sc5]), color='darkorange', linestyle='--', label='Training Max Score')

ax[0].axhline(y=max([sc1,sc2,sc3,sc4,sc5,sc6]), color='darkgreen', linestyle='--', label='Validation Max Score')

ax[0].legend(['Training score', 'Testing Score', 'Training Max Score', 'Validation Max Score'])
ax[0].set_ylabel('Accuracy Score')
ax[0].set_xlabel('Classifiers')
ax[0].set_title('Model Comparison [Accuracy Score]')

# Mean Absolute Error

ax[1].plot(['Linear\nRegression','Decision\nTree','Random\nForest', 'XG/nBoost', 'ANN', 'Ensmble\nModel'],
         [mae1, mae2, mae3, mae4, mae5, mae6],
        color = 'lightgreen')

ax[1].axhline(y=min([mae1, mae2, mae3, mae4, mae5, mae6]), color='darkgreen', linestyle='--', label='Max Score')
ax[1].legend()
ax[1].set_ylabel('Mean Absolute Error')
ax[1].set_xlabel('Classifiers')
ax[1].set_title('Model Comparison [Mean Absolute Error]')

plt.show()


# ### Submission

# In[59]:


features


# In[60]:


# scaling data
testf = scaler.transform(test[features])


# In[61]:


ypredf1 = grid_search1.predict(testf)
ypredf2 = grid_search2.predict(testf)
ypredf3 = grid_search3.predict(testf)
ypredf4 = grid_search4.predict(testf)
ypredf5 = ann.predict(testf)
ypredf5 = ypredf5.reshape(len(ypredf5),)


# In[62]:


ypredf = np.mean([ypredf2, ypredf3, ypredf4, ypredf5], axis=0)


# In[63]:


ypredf.shape, submission.shape


# In[64]:


submission['Age'] = np.round(ypredf)


# In[65]:


submission.isnull().sum()


# In[66]:


submission


# In[67]:


# Save the test dataframe with the predictions for the final sample submission
submission.to_csv('submission.csv', index=False)


# In[ ]:




